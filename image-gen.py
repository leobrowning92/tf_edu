import cairo,os,sys
from PIL import Image
import numpy as np
def imageto_array(image_path,v=False):
    im = Image.open(image_path)
    im.thumbnail((256,256))
    arrayim =  np.asarray(im)
    if v:
        print(im.size,im.format,im.mode)
        print(arrayim.shape)
        print(arrayim)
    return arrayim
def line(ctx,start,end,width=1,v=False):
    ctx.set_line_width(width)
    ctx.move_to(start[0],start[1])
    ctx.line_to(end[0],end[1])
    ctx.stroke()

def randxy(size):
    return [np.random.uniform(0,size),np.random.uniform(0,size)]
def intersect(s1,s2):
    #assert that x intervals overlap
    if max(s1[:,0])<min(s2[:,0]):
        return False # intervals do not overlap
    #gradients
    m1=(s1[0,1]-s1[1,1])/(s1[0,0]-s1[1,0])
    m2=(s2[0,1]-s2[1,1])/(s2[0,0]-s2[1,0])
    #intercepts
    b1=s1[0,1]-m1*s1[0,0]
    b2=s2[0,1]-m2*s2[0,0]
    if m1==m2:
        return False #lines are parallel
    #xi,yi on both lines
    xi=(b2-b1)/(m1-m2)
    yi=(b2*m1-b1*m2)/(m1-m2)
    if min(s1[:,0])<xi<max(s1[:,0]) and min(s2[:,0])<xi<max(s2[:,0]):
        return [xi,yi]
    else:
        return False

def make_test_image(fname,savedir,size):
    sur=cairo.ImageSurface(cairo.FORMAT_ARGB32,size,size)
    ctx=cairo.Context(sur)
    ctx.set_source_rgb(1,0,0)
    s1=np.array([randxy(size),randxy(size)])
    s2=np.array([randxy(size),randxy(size)])
    p1=intersect(s1,s2)
    line(ctx,s1[0],s1[1])
    line(ctx,s2[0],s2[1])
    if p1==False:
        pass
    else:
        ctx.set_source_rgb(0,1,0)
        ctx.rectangle(p1[0],p1[1],1,1)
        ctx.fill()
    sur.write_to_png(os.path.join(savedir,fname))
    return p1

def make_image_set(number,savedir):
    with open(savedir+"info.txt","w+") as f:
        for i in range(number):
            f.write("{0:06d}".format(i) + "," + str(make_test_image("{0:06d}".format(i) + ".png", "line_data/training", 32))+"\n")
            make_test_image("{0:06d}.png".format(i), savedir, 32 )

if __name__=="__main__":
    assert len(sys.argv)==3, "image-gen takes exactly 2 arguments, {} given.".format(len(sys.argv)-1)
    make_image_set(int(sys.argv[1]),sys.argv[2])



# imageto_array("test.png",v=True)