"""Train and run a compact residual network conditioned on deterministic MCFT."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader,Dataset


class ResidualBlock(nn.Module):
    def __init__(self,width):
        super().__init__();self.body=nn.Sequential(nn.Conv2d(width,width,3,1,1),nn.GELU(),nn.Conv2d(width,width,3,1,1))
    def forward(self,x):return x+0.2*self.body(x)


class MCFTNet(nn.Module):
    def __init__(self,width=32,blocks=6):
        super().__init__();self.head=nn.Conv2d(8,width,3,1,1)
        self.body=nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)])
        self.tail=nn.Conv2d(width,3,3,1,1);nn.init.zeros_(self.tail.weight);nn.init.zeros_(self.tail.bias)
    def forward(self,x,base):return torch.clamp(base+20.0*torch.tanh(self.tail(self.body(self.head(x)))),0,1)


class SpatialMCFTNet(nn.Module):
    """Per-frame spatial reconstruction with gated anchor-detail fusion."""
    def __init__(self,width=32,blocks=6):
        super().__init__();branch_blocks=max(1,blocks//2);fusion_blocks=max(1,blocks-branch_blocks)
        self.spatial_head=nn.Conv2d(3,width,3,1,1)
        self.spatial_body=nn.Sequential(*[ResidualBlock(width) for _ in range(branch_blocks)])
        self.anchor_head=nn.Conv2d(5,width,3,1,1)
        self.anchor_body=nn.Sequential(*[ResidualBlock(width) for _ in range(branch_blocks)])
        self.gate=nn.Sequential(nn.Conv2d(2*width,width,1),nn.GELU(),nn.Conv2d(width,width,1),nn.Sigmoid())
        self.fusion=nn.Sequential(*[ResidualBlock(width) for _ in range(fusion_blocks)])
        self.tail=nn.Conv2d(width,3,3,1,1);nn.init.zeros_(self.tail.weight);nn.init.zeros_(self.tail.bias)
    def forward(self,x,base):
        spatial=self.spatial_body(self.spatial_head(base))
        # Signed anchor proposal, proposal magnitude, and anchor distance.  The
        # spatial path never depends on the motion-propagated candidate.
        anchor_input=torch.cat((x[:,3:6]-base,x[:,6:8]),dim=1)
        anchor=self.anchor_body(self.anchor_head(anchor_input))
        gate=self.gate(torch.cat((spatial,anchor),dim=1))
        residual=0.25*torch.tanh(self.tail(self.fusion(spatial+gate*anchor)))
        return torch.clamp(base+residual,0,1)


class ProtectedMCFTNet(SpatialMCFTNet):
    """Confidence-gated correction around deterministic MCFT, not bicubic."""
    def __init__(self,width=32,blocks=6):
        super().__init__(width,blocks)
        self.output_gate=nn.Sequential(nn.Conv2d(width,1,1),nn.Sigmoid())
        nn.init.zeros_(self.output_gate[0].weight)
        nn.init.constant_(self.output_gate[0].bias,-2.0)
    def forward(self,x,base):
        protected=x[:,3:6]
        spatial=self.spatial_body(self.spatial_head(base))
        anchor_input=torch.cat((protected-base,x[:,6:8]),dim=1)
        anchor=self.anchor_body(self.anchor_head(anchor_input))
        gate=self.gate(torch.cat((spatial,anchor),dim=1))
        features=self.fusion(spatial+gate*anchor)
        correction=0.10*torch.tanh(self.tail(features))
        return torch.clamp(protected+self.output_gate(features)*correction,0,1)


class ConfidenceMCFTNet(ProtectedMCFTNet):
    """Protected refiner with explicit observation/edge/texture confidence cues."""
    def __init__(self,width=32,blocks=6):
        super().__init__(width,blocks)
        self.anchor_head=nn.Conv2d(8,width,3,1,1)
    def forward(self,x,base):
        protected=x[:,3:6]
        spatial=self.spatial_body(self.spatial_head(base))
        anchor_input=torch.cat((protected-base,x[:,6:11]),dim=1)
        anchor=self.anchor_body(self.anchor_head(anchor_input))
        gate=self.gate(torch.cat((spatial,anchor),dim=1))
        features=self.fusion(spatial+gate*anchor)
        correction=0.10*torch.tanh(self.tail(features))
        return torch.clamp(protected+self.output_gate(features)*correction,0,1)


def build_model(architecture,width,blocks):
    if architecture=='spatial':return SpatialMCFTNet(width,blocks)
    if architecture=='protected':return ProtectedMCFTNet(width,blocks)
    if architecture=='confidence':return ConfidenceMCFTNet(width,blocks)
    if architecture=='legacy':return MCFTNet(width,blocks)
    raise ValueError(f'Unknown architecture: {architecture}')


def distance_map(index,anchors,h,w):return np.full((h,w,1),min(abs(index-a) for a in anchors)/20.0,np.float32)


def make_input(base,candidate,index,anchors):
    b=base.astype(np.float32)/255;c=candidate.astype(np.float32)/255
    delta=np.mean(np.abs(c-b),axis=2,keepdims=True)
    return np.concatenate([b,c,delta,distance_map(index,anchors,*b.shape[:2])],axis=2)


def make_confidence_input(base,candidate,lr,index,anchors):
    feature=make_input(base,candidate,index,anchors)
    h,w=base.shape[:2]
    observed=lr.astype(np.float32)/255
    predicted=cv2.resize(candidate.astype(np.float32)/255,(lr.shape[1],lr.shape[0]),interpolation=cv2.INTER_AREA)
    observation=np.mean(np.abs(predicted-observed),axis=2,keepdims=True)
    observation=cv2.resize(observation,(w,h),interpolation=cv2.INTER_LINEAR)[...,None]
    by=cv2.cvtColor(base,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    cy=cv2.cvtColor(candidate,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    edge=np.abs(cv2.Laplacian(cy,cv2.CV_32F)-cv2.Laplacian(by,cv2.CV_32F))[...,None]
    mean=cv2.boxFilter(by,cv2.CV_32F,(7,7));variance=np.maximum(cv2.boxFilter(by*by,cv2.CV_32F,(7,7))-mean*mean,0)
    texture=np.sqrt(variance)[...,None]
    return np.concatenate((feature,observation,edge,texture),axis=2)


def read_base_frame(base_cap, lr_frame, index, width, height):
    if base_cap is None:
        return cv2.resize(lr_frame,(width,height),interpolation=cv2.INTER_CUBIC)
    base_cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,base=base_cap.read()
    if not ok:raise RuntimeError(f'Cannot read base frame {index}')
    if base.shape[1]!=width or base.shape[0]!=height:
        base=cv2.resize(base,(width,height),interpolation=cv2.INTER_CUBIC)
    return base


class PatchDataset(Dataset):
    def __init__(self,inputs,targets,observations):self.inputs=inputs;self.targets=targets;self.observations=observations
    def __len__(self):return len(self.inputs)
    def __getitem__(self,i):
        x=torch.from_numpy(self.inputs[i]).permute(2,0,1).float()
        y=torch.from_numpy(self.targets[i]).permute(2,0,1).float()/255
        lr=torch.from_numpy(self.observations[i]).permute(2,0,1).float()/255
        return x,y,lr


def collect(gt_path,lr_path,candidate_path,anchor_path,patch,per_frame,seed,max_frames=0,architecture='legacy',base_path=None):
    anchors=sorted(json.loads(Path(anchor_path).read_text())['anchor_indices']);aset=set(anchors)
    caps=[cv2.VideoCapture(p) for p in (gt_path,lr_path,candidate_path)];rng=np.random.default_rng(seed)
    base_cap=cv2.VideoCapture(base_path) if base_path else None
    if base_path and not base_cap.isOpened():raise RuntimeError(f'Could not open base video {base_path}')
    inputs=[];targets=[];observations=[];index=0
    while True:
        reads=[cap.read() for cap in caps]
        if not all(ok for ok,_ in reads) or (max_frames and index>=max_frames):break
        gt,lr,candidate=[frame for _,frame in reads]
        if index not in aset:
            base=read_base_frame(base_cap,lr,index,gt.shape[1],gt.shape[0])
            feature=(make_confidence_input(base,candidate,lr,index,anchors)
                     if architecture=='confidence' else make_input(base,candidate,index,anchors))
            for _ in range(per_frame):
                x=int(rng.integers(0,(gt.shape[1]-patch)//2+1))*2;y=int(rng.integers(0,(gt.shape[0]-patch)//2+1))*2
                inputs.append(feature[y:y+patch,x:x+patch].astype(np.float16));targets.append(gt[y:y+patch,x:x+patch])
                observations.append(lr[y//2:(y+patch)//2,x//2:(x+patch)//2])
        index+=1
    [cap.release() for cap in caps]
    if base_cap:base_cap.release()
    return PatchDataset(inputs,targets,observations)


def gradient_loss(a,b):
    return (torch.mean(torch.abs((a[:,:,:,1:]-a[:,:,:,:-1])-(b[:,:,:,1:]-b[:,:,:,:-1])))+
            torch.mean(torch.abs((a[:,:,1:,:]-a[:,:,:-1,:])-(b[:,:,1:,:]-b[:,:,:-1,:]))))


def device_for(name):
    if name=='auto':return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    return torch.device(name)


def train(args):
    torch.manual_seed(7);random.seed(7);device=device_for(args.device)
    trainset=collect(args.train_gt,args.train_lr,args.train_candidate,args.train_anchors,args.patch,args.patches_per_frame,7,args.max_train_frames,args.architecture,args.train_base)
    valset=collect(args.val_gt,args.val_lr,args.val_candidate,args.val_anchors,args.patch,args.val_patches_per_frame,19,args.max_val_frames,args.architecture,args.val_base)
    loader=DataLoader(trainset,batch_size=args.batch_size,shuffle=True,num_workers=0);vloader=DataLoader(valset,batch_size=args.batch_size,shuffle=False,num_workers=0)
    model=build_model(args.architecture,args.width,args.blocks).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    best=float('inf')
    for epoch in range(args.epochs):
        model.train();running=0
        for x,y,lr in loader:
            x=x.to(device);y=y.to(device);lr=lr.to(device);base=x[:,:3]
            prediction=model(x,base);charb=torch.mean(torch.sqrt((prediction-y)**2+1e-6))
            grad=gradient_loss(prediction,y);observation=torch.mean(torch.abs(torch.nn.functional.avg_pool2d(prediction,2)-lr))
            protected=x[:,3:6] if args.architecture in ('protected','confidence') else base
            residual=torch.mean(torch.abs(prediction-protected))
            # A patch guard discourages corrections whose mean error is worse
            # than the already strong deterministic MCFT proposal.
            pred_patch=torch.mean(torch.abs(prediction-y),dim=(1,2,3))
            protected_patch=torch.mean(torch.abs(protected-y),dim=(1,2,3))
            guard=torch.mean(torch.relu(pred_patch-protected_patch))
            loss=charb+0.15*grad+0.20*observation+0.005*residual+0.25*guard
            optimizer.zero_grad();loss.backward();optimizer.step();running+=float(loss.detach())*len(x)
        model.eval();validation=0
        with torch.inference_mode():
            for x,y,lr in vloader:
                x=x.to(device);y=y.to(device);prediction=model(x,x[:,:3]);validation+=float(torch.mean(torch.abs(prediction-y)))*len(x)
        validation/=len(valset);print('epoch',epoch+1,'train',running/len(trainset),'val_l1',validation)
        if validation<best:
            best=validation;torch.save({'state':model.state_dict(),'width':args.width,'blocks':args.blocks,'architecture':args.architecture,'val_l1':best},args.output)


def tiled(model,feature,device,tile,overlap):
    h,w=feature.shape[:2];acc=np.zeros((h,w,3),np.float32);weight=np.zeros((h,w,1),np.float32);step=tile-overlap
    if tile<=0:
        with torch.inference_mode():
            tensor=torch.from_numpy(feature).permute(2,0,1)[None].float().to(device)
            return np.clip(model(tensor,tensor[:,:3])[0].permute(1,2,0).cpu().numpy()*255,0,255).astype(np.uint8)
    ys=list(range(0,max(h-tile,0)+1,step));xs=list(range(0,max(w-tile,0)+1,step))
    if not ys or ys[-1]!=h-tile:ys.append(max(h-tile,0))
    if not xs or xs[-1]!=w-tile:xs.append(max(w-tile,0))
    window=np.outer(np.hanning(tile),np.hanning(tile)).astype(np.float32)[...,None];window=np.maximum(window,1e-3)
    with torch.inference_mode():
        for y in ys:
            for x in xs:
                crop=feature[y:y+tile,x:x+tile];tensor=torch.from_numpy(crop).permute(2,0,1)[None].float().to(device)
                out=model(tensor,tensor[:,:3])[0].permute(1,2,0).cpu().numpy();acc[y:y+tile,x:x+tile]+=out*window;weight[y:y+tile,x:x+tile]+=window
    return np.clip(acc/np.maximum(weight,1e-6)*255,0,255).astype(np.uint8)


def infer(args):
    device=device_for(args.device);checkpoint=torch.load(args.model,map_location=device,weights_only=True);model=build_model(checkpoint.get('architecture','legacy'),checkpoint['width'],checkpoint['blocks']).to(device);model.load_state_dict(checkpoint['state']);model.eval()
    anchors=sorted(json.loads(Path(args.anchors).read_text())['anchor_indices']);aset=set(anchors);candidate=cv2.VideoCapture(args.candidate);lr=cv2.VideoCapture(args.low_resolution)
    base_cap=cv2.VideoCapture(args.base_video) if args.base_video else None
    if args.base_video and not base_cap.isOpened():raise RuntimeError(f'Could not open base video {args.base_video}')
    n=int(candidate.get(cv2.CAP_PROP_FRAME_COUNT));n=min(n,args.max_frames) if args.max_frames else n;w=int(candidate.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(candidate.get(cv2.CAP_PROP_FRAME_HEIGHT));fps=candidate.get(cv2.CAP_PROP_FPS);writer=cv2.VideoWriter(args.output,cv2.VideoWriter_fourcc(*'FFV1'),fps,(w,h))
    if args.tile<=0 and args.infer_batch_size>1:
        for start in range(0,n,args.infer_batch_size):
            records=[];features=[];slots=[]
            for index in range(start,min(start+args.infer_batch_size,n)):
                ok,c=candidate.read();okl,l=lr.read()
                if not(ok and okl):raise RuntimeError(index)
                records.append(c)
                if index not in aset:
                    base=read_base_frame(base_cap,l,index,w,h)
                    feature=(make_confidence_input(base,c,l,index,anchors)
                             if checkpoint.get('architecture')=='confidence' else make_input(base,c,index,anchors))
                    features.append(feature);slots.append(index-start)
            if features:
                tensor=torch.from_numpy(np.stack(features)).permute(0,3,1,2).float().to(device)
                with torch.inference_mode():batch=model(tensor,tensor[:,:3]).permute(0,2,3,1).cpu().numpy()
                for slot,out in zip(slots,batch):records[slot]=np.clip(out*255,0,255).astype(np.uint8)
            for offset,frame in enumerate(records):
                writer.write(frame);print(f'[{start+offset+1}/{n}]')
        candidate.release();lr.release();writer.release()
        if base_cap:base_cap.release()
        return
    for index in range(n):
        ok,c=candidate.read();okl,l=lr.read()
        if not(ok and okl):raise RuntimeError(index)
        if index not in aset:
            base=read_base_frame(base_cap,l,index,w,h)
            feature=(make_confidence_input(base,c,l,index,anchors)
                     if checkpoint.get('architecture')=='confidence' else make_input(base,c,index,anchors))
            c=tiled(model,feature,device,args.tile,args.overlap)
        writer.write(c);print(f'[{index+1}/{n}]')
    candidate.release();lr.release();writer.release()
    if base_cap:base_cap.release()


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train')
    for prefix in ('train','val'):
        for name in ('gt','lr','candidate','anchors'):t.add_argument(f'--{prefix}-{name}',required=True)
        t.add_argument(f'--{prefix}-base',help='Optional HR-size SR/base-prior video for modern-base training.')
    t.add_argument('--output',required=True);t.add_argument('--patch',type=int,default=96);t.add_argument('--patches-per-frame',type=int,default=6);t.add_argument('--val-patches-per-frame',type=int,default=2);t.add_argument('--epochs',type=int,default=8);t.add_argument('--batch-size',type=int,default=8);t.add_argument('--width',type=int,default=32);t.add_argument('--blocks',type=int,default=6);t.add_argument('--architecture',choices=('legacy','spatial','protected','confidence'),default='legacy');t.add_argument('--lr',type=float,default=2e-4);t.add_argument('--device',default='auto');t.add_argument('--max-train-frames',type=int,default=0);t.add_argument('--max-val-frames',type=int,default=0)
    i=sub.add_parser('infer');i.add_argument('--model',required=True);i.add_argument('--candidate',required=True);i.add_argument('--low-resolution',required=True);i.add_argument('--anchors',required=True);i.add_argument('--output',required=True);i.add_argument('--base-video',help='Optional HR-size SR/base-prior video. Defaults to bicubic upsampling of LR.');i.add_argument('--tile',type=int,default=256,help='Tile size; 0 runs the full frame at once');i.add_argument('--overlap',type=int,default=32);i.add_argument('--device',default='auto');i.add_argument('--max-frames',type=int,default=0,help='Optional inference smoke-test limit');i.add_argument('--infer-batch-size',type=int,default=1,help='Full-frame inference batch size (tile must be 0)')
    args=p.parse_args();train(args) if args.command=='train' else infer(args)


if __name__=='__main__':main()
