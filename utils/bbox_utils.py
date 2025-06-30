def get_centre(bbox):
    x1,y1,x2,y2 = bbox
    return (x1+x2)/2, (y1+y2)/2

def get_width(bbox):
    return bbox[2] - bbox[0]



     

 