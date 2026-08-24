"""Domain-preserving MCFT fine-tuning with 75% REDS / 25% Vimeo replay."""
from __future__ import annotations
import argparse,json,math,os,random,sys
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from mcft_net import build_model,collect,device_for,gradient_loss,make_confidence_input,read_base_frame
from mcft_hf_stream import VimeoMCFTStream


class PairDataset(Dataset):
    def __init__(self,items):self.items=items
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        result=[]
        for feature,target,lr in self.items[i]:
            result.extend((torch.from_numpy(feature).permute(2,0,1).float(),
                           torch.from_numpy(target.copy()).permute(2,0,1).float()/255,
                           torch.from_numpy(lr.copy()).permute(2,0,1).float()/255))
        return tuple(result)


def collect_pairs(gt_path,lr_path,candidate_path,anchor_path,patch,per_pair,seed,max_frames=0,base_path=None):
    anchors=sorted(json.loads(Path(anchor_path).read_text())['anchor_indices']);aset=set(anchors)
    caps=[cv2.VideoCapture(p) for p in (gt_path,lr_path,candidate_path)];rng=np.random.default_rng(seed)
    base_cap=cv2.VideoCapture(base_path) if base_path else None
    if base_path and not base_cap.isOpened():raise RuntimeError(f'Could not open base video {base_path}')
    items=[];previous=None;index=0
    while True:
        reads=[cap.read() for cap in caps]
        if not all(ok for ok,_ in reads) or (max_frames and index>=max_frames):break
        gt,lr,candidate=[frame for _,frame in reads]
        if index not in aset:
            base=read_base_frame(base_cap,lr,index,gt.shape[1],gt.shape[0])
            current=(make_confidence_input(base,candidate,lr,index,anchors),gt,lr,index)
            if previous is not None and previous[3]==index-1:
                h,w=gt.shape[:2]
                for _ in range(per_pair):
                    x=int(rng.integers(0,(w-patch)//2+1))*2;y=int(rng.integers(0,(h-patch)//2+1))*2
                    pair=[]
                    for feature,target,obs,_ in (previous,current):
                        pair.append((feature[y:y+patch,x:x+patch].astype(np.float16),
                                     target[y:y+patch,x:x+patch],
                                     obs[y//2:(y+patch)//2,x//2:(x+patch)//2]))
                    items.append(tuple(pair))
            previous=current
        else:previous=None
        index+=1
    [cap.release() for cap in caps]
    if base_cap:base_cap.release()
    if not items:raise RuntimeError('No adjacent non-anchor REDS pairs were collected')
    return PairDataset(items)


def laplacian_overshoot(prediction,target):
    weights=prediction.new_tensor([0.114,0.587,0.299]).view(1,3,1,1)
    kernel=prediction.new_tensor([[0,1,0],[1,-4,1],[0,1,0]]).view(1,1,3,3)
    py=(prediction*weights).sum(1,keepdim=True);ty=(target*weights).sum(1,keepdim=True)
    pl=torch.nn.functional.conv2d(py,kernel,padding=1);tl=torch.nn.functional.conv2d(ty,kernel,padding=1)
    return torch.mean(torch.relu(torch.abs(pl)-torch.abs(tl)))


def texture_frequency_loss(prediction,target):
    """Texture-weighted Laplacian residual loss over luma in [0, 1]."""
    weights=prediction.new_tensor([0.114,0.587,0.299]).view(1,3,1,1)
    laplacian=prediction.new_tensor([[0,1,0],[1,-4,1],[0,1,0]]).view(1,1,3,3)
    sobel_x=prediction.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).view(1,1,3,3)/8
    sobel_y=prediction.new_tensor([[-1,-2,-1],[0,0,0],[1,2,1]]).view(1,1,3,3)/8
    py=(prediction*weights).sum(1,keepdim=True);ty=(target*weights).sum(1,keepdim=True)
    pl=torch.nn.functional.conv2d(py,laplacian,padding=1)
    tl=torch.nn.functional.conv2d(ty,laplacian,padding=1)
    tx=torch.nn.functional.conv2d(ty,sobel_x,padding=1)
    ty_edge=torch.nn.functional.conv2d(ty,sobel_y,padding=1)
    texture=torch.sqrt(tx*tx+ty_edge*ty_edge+1e-6)
    texture=texture/(torch.mean(texture,dim=(2,3),keepdim=True)+1e-6)
    texture=torch.clamp(texture,0.25,3.0)
    return torch.mean(texture*torch.abs(pl-tl))


def local_ssim_loss(prediction,target,window=7):
    """Differentiable local SSIM loss over BGR channels in [0, 1]."""
    pool=lambda value:torch.nn.functional.avg_pool2d(value,window,1,window//2)
    mu_p=pool(prediction);mu_t=pool(target)
    var_p=torch.clamp(pool(prediction*prediction)-mu_p*mu_p,min=0)
    var_t=torch.clamp(pool(target*target)-mu_t*mu_t,min=0)
    covariance=pool(prediction*target)-mu_p*mu_t
    c1=0.01**2;c2=0.03**2
    ssim=((2*mu_p*mu_t+c1)*(2*covariance+c2))/((mu_p*mu_p+mu_t*mu_t+c1)*(var_p+var_t+c2)+1e-8)
    return 1-torch.mean(torch.clamp(ssim,-1,1))


def frame_loss(model,teacher,x,target,lr,args):
    base=x[:,:3];protected=x[:,3:6];prediction=model(x,base)
    charb=torch.mean(torch.sqrt((prediction-target)**2+1e-6))
    grad=gradient_loss(prediction,target)
    observation=torch.mean(torch.abs(torch.nn.functional.avg_pool2d(prediction,2)-lr))
    pred_patch=torch.mean(torch.abs(prediction-target),dim=(1,2,3));prot_patch=torch.mean(torch.abs(protected-target),dim=(1,2,3))
    guard=torch.mean(torch.relu(pred_patch-prot_patch))
    with torch.inference_mode():teacher_prediction=teacher(x,base)
    teacher_error=torch.mean(torch.abs(teacher_prediction-target),dim=1,keepdim=True)
    protected_error=torch.mean(torch.abs(protected-target),dim=1,keepdim=True)
    reliable=(teacher_error<=protected_error).float()
    distill=torch.sum(torch.abs(prediction-teacher_prediction)*reliable)/(3*torch.sum(reliable)+1e-6)
    edge=laplacian_overshoot(prediction,target)
    structural=local_ssim_loss(prediction,target)
    texture=texture_frequency_loss(prediction,target)
    loss=(charb+args.gradient_weight*grad+0.20*observation+0.25*guard+
          args.distill_weight*distill+args.edge_weight*edge+
          args.ssim_weight*structural+args.texture_weight*texture)
    return prediction,loss


def l2sp(model,reference):
    return sum(torch.mean((parameter-reference[name])**2) for name,parameter in model.named_parameters())


def optimize_single(model,teacher,optimizer,batch,reference,args,device):
    x,target,lr=(value.to(device) for value in batch)
    _,loss=frame_loss(model,teacher,x,target,lr,args)
    loss=loss+args.l2sp_weight*l2sp(model,reference)
    optimizer.zero_grad();loss.backward();optimizer.step();return float(loss.detach())


def optimize_pair(model,teacher,optimizer,batch,reference,args,device):
    x0,y0,lr0,x1,y1,lr1=(value.to(device) for value in batch)
    p0,l0=frame_loss(model,teacher,x0,y0,lr0,args);p1,l1=frame_loss(model,teacher,x1,y1,lr1,args)
    r0=p0-x0[:,3:6];r1=p1-x1[:,3:6];t0=y0-x0[:,3:6];t1=y1-x1[:,3:6]
    temporal=torch.mean(torch.abs((r1-r0)-(t1-t0)))
    loss=0.5*(l0+l1)+args.temporal_weight*temporal+args.l2sp_weight*l2sp(model,reference)
    optimizer.zero_grad();loss.backward();optimizer.step();return float(loss.detach())


def validate(model,dataset,batch_size,device):
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0);total=0;count=0;model.eval()
    with torch.inference_mode():
        for x,target,_ in loader:
            x=x.to(device);target=target.to(device);prediction=model(x,x[:,:3])
            total+=float(torch.mean(torch.abs(prediction-target)))*len(x);count+=len(x)
    return total/count


def train(args):
    torch.manual_seed(args.seed);random.seed(args.seed);device=device_for(args.device)
    checkpoint=torch.load(args.init,map_location=device,weights_only=True)
    if checkpoint.get('architecture')!='confidence':raise ValueError('--init must be a confidence checkpoint')
    model=build_model('confidence',checkpoint['width'],checkpoint['blocks']).to(device);model.load_state_dict(checkpoint['state'])
    teacher=build_model('confidence',checkpoint['width'],checkpoint['blocks']).to(device);teacher.load_state_dict(checkpoint['state']);teacher.eval()
    for p in teacher.parameters():p.requires_grad_(False)
    reference={name:p.detach().clone() for name,p in model.named_parameters()}
    pairs=collect_pairs(args.reds_gt,args.reds_lr,args.reds_candidate,args.reds_anchors,args.patch,args.reds_patches_per_pair,args.seed,args.max_reds_frames,args.reds_base)
    validation=collect(args.val_gt,args.val_lr,args.val_candidate,args.val_anchors,args.patch,args.val_patches_per_frame,19,0,'confidence',args.val_base)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4);best=math.inf
    for epoch in range(args.epochs):
        reds_loader=DataLoader(pairs,batch_size=args.batch_size,shuffle=True,num_workers=0);reds_iter=iter(reds_loader)
        vimeo=VimeoMCFTStream('train',args.vimeo_triplets,args.vimeo_patches_per_triplet,args.patch,args.seed+epoch,args.shuffle_buffer,args.flow_method,args.proposal_strength)
        vimeo_loader=DataLoader(vimeo,batch_size=args.batch_size,num_workers=0);model.train();running=[];step=0
        for vimeo_batch in vimeo_loader:
            for _ in range(args.reds_steps_per_vimeo):
                try:reds_batch=next(reds_iter)
                except StopIteration:reds_iter=iter(reds_loader);reds_batch=next(reds_iter)
                running.append(optimize_pair(model,teacher,optimizer,reds_batch,reference,args,device));step+=1
            running.append(optimize_single(model,teacher,optimizer,vimeo_batch,reference,args,device));step+=1
            if args.log_every and step%args.log_every==0:print('epoch',epoch+1,'step',step,'train',np.mean(running[-args.log_every:]),flush=True)
        score=validate(model,validation,args.batch_size,device);print('epoch',epoch+1,'train',np.mean(running),'reds006_val_l1',score,flush=True)
        if score<best:
            best=score;torch.save({'state':model.state_dict(),'width':checkpoint['width'],'blocks':checkpoint['blocks'],'architecture':'confidence','val_l1':best,'training':'reds75_vimeo25_replay','teacher':args.init,'vimeo_triplets':args.vimeo_triplets},args.output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('init','output','reds-gt','reds-lr','reds-candidate','reds-anchors','val-gt','val-lr','val-candidate','val-anchors'):p.add_argument('--'+name,required=True)
    p.add_argument('--reds-base',help='Optional HR-size modern SR/base-prior video for REDS replay.')
    p.add_argument('--val-base',help='Optional HR-size modern SR/base-prior video for validation.')
    p.add_argument('--epochs',type=int,default=1);p.add_argument('--vimeo-triplets',type=int,default=100);p.add_argument('--vimeo-patches-per-triplet',type=int,default=2);p.add_argument('--reds-patches-per-pair',type=int,default=2);p.add_argument('--reds-steps-per-vimeo',type=int,default=3)
    p.add_argument('--patch',type=int,default=96);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--val-patches-per-frame',type=int,default=4);p.add_argument('--max-reds-frames',type=int,default=0)
    p.add_argument('--lr',type=float,default=5e-5);p.add_argument('--distill-weight',type=float,default=0.10);p.add_argument('--edge-weight',type=float,default=0.05);p.add_argument('--ssim-weight',type=float,default=0.05);p.add_argument('--gradient-weight',type=float,default=0.15);p.add_argument('--texture-weight',type=float,default=0.0);p.add_argument('--temporal-weight',type=float,default=0.10);p.add_argument('--l2sp-weight',type=float,default=0.01)
    p.add_argument('--shuffle-buffer',type=int,default=16);p.add_argument('--flow-method',choices=('farneback','dis'),default='dis');p.add_argument('--proposal-strength',type=float,default=0.5);p.add_argument('--log-every',type=int,default=8);p.add_argument('--seed',type=int,default=7);p.add_argument('--device',default='auto')
    train(p.parse_args())


if __name__=='__main__':
    main()
    # Hugging Face's macOS HTTP worker may retry after an exhausted iterable.
    # torch.save is synchronous, so flush completed metrics and exit cleanly.
    sys.stdout.flush();sys.stderr.flush();os._exit(0)
