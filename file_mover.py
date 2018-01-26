from shutil import copyfile

for dst_i in range(100):
    src_i = dst_i//5
    copyfile("data/bar-3d/validation/graph_{}.png".format(src_i),
             "/home/will//Documents/D3m/4yp-slide/prior/{}.png".format(dst_i))
