"""Lossless controlled MCFT with gradient-domain, observation-constrained reconstruction."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import cv2
import numpy as np
from skimage.metrics import structural_similarity
from motion_frequency_transfer import compute_flow, flow_confidence, remap_with_displacement, resize_flow


def local_subpixel_refine(target: np.ndarray, warped: np.ndarray, confidence: np.ndarray,
                          patch: int = 128, stride: int = 64,
                          max_shift: float = 2.0, min_response: float = 0.08):
    h, w = target.shape; win = cv2.createHanningWindow((patch, patch), cv2.CV_32F)
    accum = np.zeros_like(target, np.float32); weights = np.zeros_like(target, np.float32)
    refined_conf = np.zeros_like(target, np.float32)
    ys = list(range(0, h-patch+1, stride)); xs = list(range(0, w-patch+1, stride))
    if ys[-1] != h-patch: ys.append(h-patch)
    if xs[-1] != w-patch: xs.append(w-patch)
    for y in ys:
        for x in xs:
            t = target[y:y+patch, x:x+patch].astype(np.float32)
            a = warped[y:y+patch, x:x+patch].astype(np.float32)
            shift, response = cv2.phaseCorrelate(t, a, win)
            dx, dy = np.clip(shift, -max_shift, max_shift)
            if response < min_response:
                dx = dy = 0.0
            yy, xx = np.mgrid[:patch, :patch]
            aligned = cv2.remap(a, (xx + dx).astype(np.float32), (yy + dy).astype(np.float32),
                                cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)
            weight = win * max(0.0, min(1.0, response)) * confidence[y:y+patch, x:x+patch]
            accum[y:y+patch, x:x+patch] += aligned * weight
            refined_conf[y:y+patch, x:x+patch] += weight
            weights[y:y+patch, x:x+patch] += win
    fallback = warped
    refined = np.where(refined_conf > 1e-5, accum / np.maximum(refined_conf, 1e-5), fallback)
    return refined.astype(np.float32), np.clip(refined_conf / np.maximum(weights, 1e-5), 0, 1)


def local_affine_refine(target: np.ndarray, warped: np.ndarray, confidence: np.ndarray,
                        patch: int = 128, stride: int = 64):
    """Locally refine flow-warped patches with conservative affine ECC."""
    h,w=target.shape; win=cv2.createHanningWindow((patch,patch),cv2.CV_32F)
    accum=np.zeros_like(target,np.float32); weights=np.zeros_like(target,np.float32)
    conf_accum=np.zeros_like(target,np.float32)
    ys=list(range(0,h-patch+1,stride)); xs=list(range(0,w-patch+1,stride))
    if ys[-1]!=h-patch: ys.append(h-patch)
    if xs[-1]!=w-patch: xs.append(w-patch)
    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,20,1e-4)
    for y in ys:
        for x in xs:
            template=target[y:y+patch,x:x+patch].astype(np.float32)/255.0
            source=warped[y:y+patch,x:x+patch].astype(np.float32)/255.0
            matrix=np.float32([[1,0,0],[0,1,0]])
            score=0.0
            try:
                score,matrix=cv2.findTransformECC(template,source,matrix,cv2.MOTION_AFFINE,
                                                  criteria,inputMask=None,gaussFiltSize=5)
                # Reject implausible local scale/shear that would manufacture texture.
                if np.linalg.norm(matrix[:,:2]-np.eye(2,dtype=np.float32))>0.12:
                    score=0.0; matrix=np.float32([[1,0,0],[0,1,0]])
            except cv2.error:
                pass
            aligned=cv2.warpAffine(source,matrix,(patch,patch),
                                   flags=cv2.INTER_CUBIC|cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_REFLECT101)*255.0
            weight=win*np.clip(score,0,1)*confidence[y:y+patch,x:x+patch]
            accum[y:y+patch,x:x+patch]+=aligned*weight
            conf_accum[y:y+patch,x:x+patch]+=weight
            weights[y:y+patch,x:x+patch]+=win
    refined=np.where(conf_accum>1e-5,accum/np.maximum(conf_accum,1e-5),warped)
    return refined.astype(np.float32),np.clip(conf_accum/np.maximum(weights,1e-5),0,1)


def gradients(x: np.ndarray):
    gx = np.zeros_like(x); gy = np.zeros_like(x)
    gx[:, :-1] = x[:, 1:] - x[:, :-1]
    gy[:-1, :] = x[1:, :] - x[:-1, :]
    return gx, gy


def multiscale_residual(residual: np.ndarray, confidence: np.ndarray,
                        sigma: float = 1.2) -> np.ndarray:
    """Keep medium residual detail and gate the alignment-sensitive fine band."""
    medium = cv2.GaussianBlur(residual, (0, 0), sigma)
    fine = residual - medium
    return medium + confidence * fine


def content_adaptive_residual(residual: np.ndarray, confidence: np.ndarray,
                              base: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    """Use stricter fine-band transfer in flat/unstable regions."""
    mean=cv2.boxFilter(base,cv2.CV_32F,(7,7))
    variance=np.maximum(cv2.boxFilter(base*base,cv2.CV_32F,(7,7))-mean*mean,0)
    texture=np.sqrt(variance)/(np.sqrt(variance)+8.0)
    medium=cv2.GaussianBlur(residual,(0,0),sigma);fine=residual-medium
    return (0.65+0.35*texture)*medium+confidence*texture*fine


def screened_poisson(base: np.ndarray, gx: np.ndarray, gy: np.ndarray,
                     lam: float, iterations: int):
    dtg = np.zeros_like(base)
    dtg[:, 1:] += gx[:, :-1]; dtg[:, :-1] -= gx[:, :-1]
    dtg[1:, :] += gy[:-1, :]; dtg[:-1, :] -= gy[:-1, :]
    rhs = base + lam * dtg
    x = base.copy()
    for _ in range(iterations):
        neighbors = np.zeros_like(x); degree = np.zeros_like(x)
        neighbors[:, 1:] += x[:, :-1]; degree[:, 1:] += 1
        neighbors[:, :-1] += x[:, 1:]; degree[:, :-1] += 1
        neighbors[1:, :] += x[:-1, :]; degree[1:, :] += 1
        neighbors[:-1, :] += x[1:, :]; degree[:-1, :] += 1
        x = (rhs + lam * neighbors) / (1.0 + lam * degree)
    return x


def project_observation(x: np.ndarray, observed_lr: np.ndarray, strength: float):
    h, w = observed_lr.shape
    predicted = cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA)
    residual = observed_lr - predicted
    return x + strength * cv2.resize(residual, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_CUBIC)


def project_observation_operator(x: np.ndarray, observed_lr: np.ndarray,
                                 strength: float, model: dict | None) -> np.ndarray:
    if not model:
        return project_observation(x, observed_lr, strength)
    sigma = float(model.get('blur_sigma_hr', 0.0))
    phase_x = float(model.get('phase_x_hr', 0.0)); phase_y = float(model.get('phase_y_hr', 0.0))
    work = cv2.GaussianBlur(x,(0,0),sigma) if sigma>1e-6 else x
    if abs(phase_x)+abs(phase_y)>1e-6:
        matrix=np.float32([[1,0,phase_x],[0,1,phase_y]])
        work=cv2.warpAffine(work,matrix,(x.shape[1],x.shape[0]),flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REFLECT101)
    kernel = cv2.INTER_CUBIC if model.get('resize_kernel') == 'bicubic' else cv2.INTER_AREA
    predicted=cv2.resize(work,(observed_lr.shape[1],observed_lr.shape[0]),interpolation=kernel)
    update=cv2.resize(observed_lr-predicted,(x.shape[1],x.shape[0]),interpolation=cv2.INTER_CUBIC)
    if abs(phase_x)+abs(phase_y)>1e-6:
        matrix=np.float32([[1,0,-phase_x],[0,1,-phase_y]])
        update=cv2.warpAffine(update,matrix,(x.shape[1],x.shape[0]),flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REFLECT101)
    if sigma>1e-6: update=cv2.GaussianBlur(update,(0,0),sigma)
    return x+strength*update


def multiframe_project(x: np.ndarray, target_lr: np.ndarray,
                       neighbors_lr: list[np.ndarray], strength: float) -> np.ndarray:
    """Back-project aligned neighboring LR observations into the target frame.

    Each neighbor correction is transported in both directions and accepted
    only where forward/backward flow and photometric consistency agree.
    """
    if not neighbors_lr or strength <= 0:
        return x
    height, width = x.shape
    target_bgr = cv2.cvtColor(target_lr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    correction = np.zeros_like(x, np.float32)
    weight_sum = np.zeros_like(x, np.float32)
    for neighbor_lr in neighbors_lr:
        neighbor_bgr = cv2.cvtColor(neighbor_lr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        target_to_neighbor = compute_flow(target_bgr, neighbor_bgr, 'dis')
        neighbor_to_target = compute_flow(neighbor_bgr, target_bgr, 'dis')
        confidence = flow_confidence(target_bgr, neighbor_bgr, target_to_neighbor,
                                     1.0, 0.06, 'dis')
        # Predict the neighbor observation from the current HR estimate.
        n_to_t_hr = resize_flow(neighbor_to_target, width, height)
        predicted_neighbor_hr = remap_with_displacement(
            x, n_to_t_hr, cv2.BORDER_REFLECT101, cv2.INTER_CUBIC
        )
        predicted_neighbor_lr = cv2.resize(
            predicted_neighbor_hr, (neighbor_lr.shape[1], neighbor_lr.shape[0]),
            interpolation=cv2.INTER_AREA
        )
        residual_lr = neighbor_lr.astype(np.float32) - predicted_neighbor_lr
        residual_hr = cv2.resize(residual_lr, (width, height), interpolation=cv2.INTER_CUBIC)
        t_to_n_hr = resize_flow(target_to_neighbor, width, height)
        aligned_residual = remap_with_displacement(
            residual_hr, t_to_n_hr, cv2.BORDER_REFLECT101, cv2.INTER_CUBIC
        )
        confidence_hr = cv2.resize(confidence, (width, height), interpolation=cv2.INTER_LINEAR)
        correction += confidence_hr * aligned_residual
        weight_sum += confidence_hr
    update = correction / np.maximum(weight_sum, 1e-6)
    return x + strength * np.clip(weight_sum, 0.0, 1.0) * update


def quality(gt: np.ndarray, x: np.ndarray):
    mse = np.mean((gt-x)**2)
    gx, gy = gradients(gt); xx, xy = gradients(x)
    return (10*np.log10(255**2/max(float(mse),1e-12)),
            structural_similarity(gt, x, data_range=255),
            float(np.mean(np.abs(np.hypot(gx,gy)-np.hypot(xx,xy)))))


def read_at(cap, i):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
    if not ok: raise RuntimeError(f"Cannot read frame {i}")
    return f


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ground-truth',required=True); p.add_argument('--anchors',required=True)
    p.add_argument('--low-resolution', help='Actual LR video. If omitted, LR is simulated losslessly from ground truth.')
    p.add_argument('--output-video', help='Write a complete MCFT video (anchors are copied from ground truth).')
    p.add_argument('--save-frames', action='store_true', help='Save lossless GT/bicubic/MCFT PNG triplets.')
    p.add_argument('--output-dir',default='constrained_mcft'); p.add_argument('--stride',type=int,default=10)
    p.add_argument('--max-frames',type=int,default=30); p.add_argument('--lambda-gradient',type=float,default=.35)
    p.add_argument('--iterations',type=int,default=35); p.add_argument('--projection-strength',type=float,default=.8)
    p.add_argument('--gradient-source', choices=['full','residual-multiscale'], default='full')
    p.add_argument('--chroma-strength', type=float, default=0.0)
    p.add_argument('--multiframe-radius', type=int, default=0)
    p.add_argument('--multiframe-strength', type=float, default=0.2)
    p.add_argument('--degradation-model')
    p.add_argument('--flow-method', choices=['dis','farneback','raft'], default='dis')
    p.add_argument('--local-refinement', choices=['phase','affine','none'], default='phase')
    p.add_argument('--content-adaptive', action='store_true')
    args=p.parse_args(); manifest=json.loads(Path(args.anchors).read_text()); anchors=sorted(map(int,manifest['anchor_indices']))
    degradation=json.loads(Path(args.degradation_model).read_text()) if args.degradation_model else None
    cap=cv2.VideoCapture(args.ground_truth); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); aset=set(anchors)
    lr_cap=cv2.VideoCapture(args.low_resolution) if args.low_resolution else None
    fps=cap.get(cv2.CAP_PROP_FPS); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # Anchors are reused many times; cache them once instead of repeatedly seeking
    # through the compressed source video.
    anchor_cache={index:read_at(cap,index) for index in anchors}
    writer=None
    if args.output_video:
        codec='FFV1' if Path(args.output_video).suffix.lower() in {'.mkv','.avi'} else 'mp4v'
        writer=cv2.VideoWriter(args.output_video,cv2.VideoWriter_fourcc(*codec),fps,(W,H))
        if not writer.isOpened(): raise RuntimeError(f'Cannot open output video {args.output_video}')
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); rows=[]
    mf_dot=0.0; mf_energy=0.0
    if writer:
        targets=list(range(n))
    else:
        targets=[i for i in range(0,n,args.stride) if i not in aset][:args.max_frames]
    for k,i in enumerate(targets):
        gt=read_at(cap,i)
        if i in aset and writer:
            writer.write(gt)
            print(f'[{k+1}/{len(targets)}] frame {i} anchor')
            continue
        observed=read_at(lr_cap,i) if lr_cap else cv2.resize(gt,(W//2,H//2),interpolation=cv2.INTER_AREA)
        base=cv2.resize(observed,(W,H),interpolation=cv2.INTER_CUBIC)
        tg=cv2.cvtColor(observed,cv2.COLOR_BGR2GRAY); by=cv2.cvtColor(base,cv2.COLOR_BGR2YCrCb)[...,0].astype(np.float32)
        pos=np.searchsorted(anchors,i); near=anchors[max(0,pos-1):min(len(anchors),pos+1)]
        gradx_sum=np.zeros_like(by); grady_sum=np.zeros_like(by); weight_sum=np.zeros_like(by)
        chroma_sum=np.zeros((H,W,2),np.float32)
        for aidx in near:
            af=anchor_cache[aidx]
            down_kernel=(cv2.INTER_CUBIC if degradation and degradation.get('resize_kernel')=='bicubic'
                         else cv2.INTER_AREA)
            alr=cv2.resize(af,(W//2,H//2),interpolation=down_kernel)
            flow=compute_flow(cv2.cvtColor(tg,cv2.COLOR_GRAY2BGR),alr,args.flow_method)
            conf=flow_confidence(cv2.cvtColor(tg,cv2.COLOR_GRAY2BGR),alr,flow,1.5,.08,args.flow_method)
            flowh=resize_flow(flow,W,H); confh=cv2.resize(conf,(W,H),interpolation=cv2.INTER_LINEAR)
            ay=cv2.cvtColor(af,cv2.COLOR_BGR2YCrCb)[...,0].astype(np.float32)
            warped=remap_with_displacement(ay,flowh,cv2.BORDER_REFLECT101,cv2.INTER_CUBIC)
            if args.local_refinement=='affine': refined,rc=local_affine_refine(by,warped,confh)
            elif args.local_refinement=='none': refined,rc=warped,confh
            else: refined,rc=local_subpixel_refine(by,warped,confh)
            wgt=rc*np.exp(-abs(i-aidx)/7.0)
            if args.gradient_source=='residual-multiscale':
                anchor_base=cv2.resize(alr,(W,H),interpolation=cv2.INTER_CUBIC)
                anchor_base_y=cv2.cvtColor(anchor_base,cv2.COLOR_BGR2YCrCb)[...,0].astype(np.float32)
                residual=ay-anchor_base_y
                warped_residual=remap_with_displacement(residual,flowh,cv2.BORDER_REFLECT101,cv2.INTER_CUBIC)
                transferred=(content_adaptive_residual(warped_residual,rc,by)
                             if args.content_adaptive else multiscale_residual(warped_residual,rc))
                ax,ayg=gradients(by+transferred)
                if args.chroma_strength>0:
                    anchor_cc=cv2.cvtColor(af,cv2.COLOR_BGR2YCrCb)[...,1:].astype(np.float32)
                    base_cc=cv2.cvtColor(anchor_base,cv2.COLOR_BGR2YCrCb)[...,1:].astype(np.float32)
                    cc_res=remap_with_displacement(anchor_cc-base_cc,flowh,cv2.BORDER_REFLECT101,cv2.INTER_CUBIC)
                    chroma_sum+=wgt[...,None]*cc_res
            else:
                ax,ayg=gradients(refined)
            gradx_sum+=wgt*ax; grady_sum+=wgt*ayg; weight_sum+=wgt
        bx,bgy=gradients(by); c=np.clip(weight_sum,0,1)
        gx=(1-c)*bx+c*gradx_sum/np.maximum(weight_sum,1e-6)
        gy=(1-c)*bgy+c*grady_sum/np.maximum(weight_sum,1e-6)
        rec=screened_poisson(by,gx,gy,args.lambda_gradient,args.iterations)
        rec=project_observation_operator(rec,tg.astype(np.float32),args.projection_strength,degradation)
        rec_before_multiframe=rec.copy()
        if args.multiframe_radius>0 and lr_cap:
            neighbors=[]
            for offset in range(-args.multiframe_radius,args.multiframe_radius+1):
                j=i+offset
                if offset and 0<=j<n and j not in aset:
                    nf=read_at(lr_cap,j)
                    neighbors.append(cv2.cvtColor(nf,cv2.COLOR_BGR2GRAY).astype(np.float32))
            rec=multiframe_project(rec,tg.astype(np.float32),neighbors,args.multiframe_strength)
            rec=project_observation_operator(rec,tg.astype(np.float32),args.projection_strength,degradation)
        ycc=cv2.cvtColor(base,cv2.COLOR_BGR2YCrCb); ycc[...,0]=np.clip(rec,0,255).astype(np.uint8)
        if args.chroma_strength>0:
            cc_delta=chroma_sum/np.maximum(weight_sum[...,None],1e-6)
            ycc[...,1:]=np.clip(ycc[...,1:].astype(np.float32)+args.chroma_strength*c[...,None]*cc_delta,0,255).astype(np.uint8)
        result=cv2.cvtColor(ycc,cv2.COLOR_YCrCb2BGR)
        gty=cv2.cvtColor(gt,cv2.COLOR_BGR2YCrCb)[...,0].astype(np.float32)
        if args.multiframe_radius>0:
            mf_delta=rec-rec_before_multiframe
            mf_dot+=float(np.sum(mf_delta*(gty-rec_before_multiframe)))
            mf_energy+=float(np.sum(mf_delta*mf_delta))
        for name,img in [('bicubic',by),('constrained',rec)]:
            ps,ss,em=quality(gty,img); rows.append({'frame':i,'method':name,'psnr_y':ps,'ssim_y':ss,'edge_mae':em})
        if args.save_frames:
            cv2.imwrite(str(out/f'{i:04d}_gt.png'),gt)
            cv2.imwrite(str(out/f'{i:04d}_bicubic.png'),base)
            cv2.imwrite(str(out/f'{i:04d}_mcft.png'),result)
        if writer: writer.write(result)
        print(f'[{k+1}/{len(targets)}] frame {i}')
    cap.release()
    if lr_cap: lr_cap.release()
    if writer: writer.release()
    with (out/'metrics.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=rows[0]);wr.writeheader();wr.writerows(rows)
    for name in ['bicubic','constrained']:
        q=[r for r in rows if r['method']==name]
        print(name,'PSNR',np.mean([r['psnr_y'] for r in q]),'SSIM',np.mean([r['ssim_y'] for r in q]),'Edge',np.mean([r['edge_mae'] for r in q]))
    if mf_energy>0:
        print('multiframe optimal correction scale',mf_dot/mf_energy)

if __name__=='__main__':main()
