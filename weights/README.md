# Model files (not stored in git)

| File | Used by | How to get it |
|---|---|---|
| `yolov8n.pt` (~6 MB) | Detection mode | Public Ultralytics model — download it with the command below |
| `crowd_vgg19.pth` (~86 MB) | Density mode | Delivered separately by the project owner |

Download `yolov8n.pt` from the project folder (Windows 10/11 has `curl` built in):

```
curl -L -o weights\yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
```

The apps start without `crowd_vgg19.pth` and work in Detection mode. Tests that need a missing model are skipped, not failed.
