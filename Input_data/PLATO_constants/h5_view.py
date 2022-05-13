# HDF5 Visualizer
# Olivier Roth

import h5py
import argparse

def hierarchical(file, depth=0, level=16, info=2, stop=140):
    """
    Visualization of an HDF5 file.

    Parameters:
    -----------
    file: str
        File to visualize.

    Optional parameters:
    -------------------
    depth: int, default=0
        Max depth in the architecture to print. 0 for not setting a limit.
    level: int, default=10
        Max level to print. (At the same depth, how many item will be printed at most)
    info: int, default=1
        Infos on the groups
        0: no info
        1: attrs and shape of the groups
        2: attrs and content (recommended for files with only one value per group,
            like a constants file)
    stop: int, default=140
        Stop printing after `stop` lines.
        (Low for better performance as opening more groups is a costing processus.)
    """
    print("Visualization of the HDF5 file",file)
    print("-------------------------------"+"-"*len(file),"\n")

    f = h5py.File(file,'r')

    if info!=0:
        for key in f.attrs.keys():
            print(key," : ", str(f.attrs.get(key)))
            print("\n")


    datalist = []
    def returnname(name):
        if name not in datalist:
            return name
        else:
            return None
    looper = 1

    depth_i = 0
    depth_tab = []

    count=-1

    while looper == 1:
        name = f.visit(returnname)
        if name == None:
            looper = 0
            continue

        count+=1
        if count==stop:
            print("...")
            print("\n--------------------")
            print("Stop printing, too many groups. (Increase `stop` parameter to see more)")
            looper = 0
            continue

        datalist.append(name)

        name_split = name.split('/')

        depth_i_0 = depth_i # keep previous len
        depth_i = len(name_split) # current len

        if depth_i==depth_i_0:
            depth_tab.append(depth_i)

        if depth_i!=depth_i_0:
            depth_tab = [] # empty if len changes
            p3p = 0


        if len(depth_tab)+1>level:
            if p3p==0:
                print(" |\t"*(depth_i-1), "...")
            p3p = 1
            continue

        if depth!=0:
            if depth_i>depth:
                continue

        attr = []
        for key in f[name].attrs.keys():
            attr.append([key, str(f[name].attrs.get(key))])

        if isinstance(f[name], h5py.Dataset):
            if info==0:
                pass
            elif info==1:
                info_dat = "; shape=%s"%str(f[name].shape)
            elif info==2:
                info_dat = "; {}".format(f[name][()])
            else:
                raise ValueError("`info` must be 0, 1 or 2, not {}. See documentation for more details.".format(info))
        else:
            info_dat = ""

        if len(attr)==0 or info==0:
            print(" |\t"*(depth_i-1),name_split[-1],info_dat)
        else:
            print(" |\t"*(depth_i-1),name_split[-1], [f"{att[0]} : {att[1]}" for att in attr],info_dat)





if __name__ == "__main__":

    class CustomFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    parser = argparse.ArgumentParser(formatter_class=CustomFormatter)

    parser.add_argument('-f', '--file', help='hdf5 file to visualize.', required=True)
    parser.add_argument('-d','--depth', help='type: int. Max depth in the architecture to print. 0 for not setting a limit.', nargs='?', default=0, type=int)
    parser.add_argument('-l','--level', help='type: int. Max level to print. (At the same depth, how many item will be printed at most)', nargs='?', default=16, type=int)
    parser.add_argument('-i','--info', choices=[0,1,2], help="type: int. Infos on the groups. \n0: no info. \n1: attrs and shape of the groups. \n2: attrs and content (recommended for files with only one value per group, like a constants file)", nargs='?', default=2, type=int)
    parser.add_argument('-s','--stop', help="type: int. Stop printing after `stop` lines. \n(Low for better performance as opening more groups is a costing processus.)", nargs='?', default=140, type=int)

    args = parser.parse_args()

    hierarchical(args.file, depth=args.depth, level=args.level, info=args.info, stop=args.stop)
