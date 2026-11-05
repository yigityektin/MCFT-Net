import argparse
import cv2
import numpy as np
import torch

from basicsr.archs.basicvsrpp_arch import BasicVSRPlusPlus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--spynet", required=True)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise RuntimeError("No frames")

    h, w = frames[0].shape[:2]
    x = np.stack(frames).astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(0, 3, 1, 2).unsqueeze(0).cuda()

    model = BasicVSRPlusPlus(
        mid_channels=64,
        num_blocks=7,
        is_low_res_input=True,
        spynet_path=args.spynet,
        cpu_cache_length=30,
    ).cuda().eval()

    state = torch.load(args.model, map_location="cpu", weights_only=False)
    state = state.get("params", state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("missing:", len(missing), "unexpected:", len(unexpected))

    with torch.inference_mode():
        y = model(x)[0].clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()

    out = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"FFV1"),
        fps,
        (w * 2, h * 2),
    )

    for frame in y:
        frame = (frame[:, :, ::-1] * 255.0 + 0.5).astype(np.uint8)
        out.write(frame)

    out.release()
    print("wrote", args.output, len(y), "frames")


if __name__ == "__main__":
    main()
