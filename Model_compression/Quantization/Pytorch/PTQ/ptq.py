

def run(**kwargs):
    # Usage: import train; train.run(imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
