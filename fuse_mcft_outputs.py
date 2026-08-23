"""Conservatively blend a protected neural refinement with deterministic MCFT."""
import argparse
import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',required=True);p.add_argument('--refined',required=True)
    p.add_argument('--output',required=True);p.add_argument('--strength',type=float,default=0.5)
    args=p.parse_args();base=cv2.VideoCapture(args.base);refined=cv2.VideoCapture(args.refined)
    w=int(base.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(base.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=base.get(cv2.CAP_PROP_FPS);writer=cv2.VideoWriter(args.output,cv2.VideoWriter_fourcc(*'FFV1'),fps,(w,h))
    if not all((base.isOpened(),refined.isOpened(),writer.isOpened())):raise RuntimeError('Could not open videos')
    count=0
    while True:
        ok,b=base.read();okr,r=refined.read()
        if not(ok and okr):break
        out=np.clip(b.astype(np.float32)+args.strength*(r.astype(np.float32)-b),0,255).astype(np.uint8)
        writer.write(out);count+=1
    base.release();refined.release();writer.release();print(f'[OK] wrote {count} frames to {args.output}')


if __name__=='__main__':main()
