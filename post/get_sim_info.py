import os
import glob
import numpy as np

__macr_names__ = ['phi', 'uz']
__info__ = dict()

def get_filenames_macr(macr_name, path):
    file_list = sorted(glob.glob(path + "*" + macr_name + "*.bin"))
    return file_list

def get_macr_steps(path):
    file_list = get_filenames_macr(__macr_names__[0], path)
    step_set = set()
    for file in file_list:
        step_str = file.split(__macr_names__[0])[-1]
        step_str = step_str[:-4]  # remove ".bin"
        step_int = int(step_str)
        step_set.add(step_int)
    macr_steps = sorted(step_set)
    return macr_steps

def retrieve_sim_info(path):
    if len(__info__) == 0:
        filename = glob.glob(path + "*info*.txt")[0]
        with open(filename, "r") as f:
            lines = f.readlines()
            lines_trimmed = [line.strip() for line in lines]

            try:
                __info__['ID'] = [str(txt.split(" ")[-1]) for txt in lines_trimmed
                                  if 'Simulation ID' in txt][0]
            except BaseException:
                print("Not able to get ID from info file")

            try:
                __info__['Prc'] = [txt.split(" ")[-1] for txt in lines_trimmed
                                   if 'Precision' in txt][0]
            except BaseException:
                print("Not able to get Precision from info file")

            try:
                __info__['NX'] = [int(txt.split(" ")[-1]) for txt in lines_trimmed
                                   if 'NX' in txt][0]
            except BaseException:
                print("Not able to get NX from info file")

            try:
                __info__['NY'] = [int(txt.split(" ")[-1]) for txt in lines_trimmed
                                   if 'NY' in txt][0]
            except BaseException:
                print("Not able to get NY from info file")

            try:
                __info__['NZ'] = [int(txt.split(" ")[-1]) for txt in lines_trimmed
                                   if 'NZ:' in txt][0]
            except BaseException:
                print("Not able to get NZ from info file")

            try:
                __info__['NZ_TOTAL'] = [int(txt.split(" ")[-1]) for txt in lines_trimmed
                                        if 'NZ_TOTAL' in txt][0]
            except BaseException:
                print("Not able to get NZ_TOTAL from info file")

            try:
                __info__['Tau'] = [float(txt.split(" ")[-1]) for txt in lines_trimmed
                                   if 'Tau' in txt][0]
            except BaseException:
                print("Not able to get Tau from info file")

            try:
                __info__['Umax'] = [float(txt.split(" ")[-1]) for txt in lines_trimmed
                                    if 'Umax' in txt][0]
            except BaseException:
                print("Not able to get Umax from info file")

            try:
                __info__['Nsteps'] = [int(txt.split(" ")[-1]) for txt in lines_trimmed
                                      if 'Nsteps' in txt][0]
            except BaseException:
                print("Not able to get Nsteps from info file")

    return __info__

def read_file_macr_3d(macr_filename, path):
    info = retrieve_sim_info(path)
    if info['Prc'] == 'double':
        dtype = 'd'
    elif info['Prc'] == 'float':
        dtype = 'f'
    with open(macr_filename, "r") as f:
        vec = np.fromfile(f, dtype)
        vec_3d = np.reshape(vec, (info['NZ_TOTAL'], info['NY'], info['NX']), 'C')
        return np.swapaxes(vec_3d, 0, 2)

def get_macrs_from_step(step, path):
    macr = dict()
    all_filenames = []

    for macr_name in __macr_names__:
        all_filenames.append(get_filenames_macr(macr_name, path))

    flat_filenames = [filename for sublist in all_filenames for filename in sublist]

    step_filenames = [
        filename for filename in flat_filenames
        if any([f"{macr}{step:06d}.bin" in filename for macr in __macr_names__])
    ]

    if len(step_filenames) == 0:
        return None

    for filename in step_filenames:
        for macr_name in __macr_names__:
            if macr_name in filename:
                macr[macr_name] = read_file_macr_3d(filename, path)

    return macr

def get_all_macrs(path):
    macr = dict()
    filenames = dict()

    for macr_name in __macr_names__:
        filenames[macr_name] = get_filenames_macr(macr_name, path)

    min_length = min(len(filenames[key]) for key in filenames)

    for i in range(min_length):
        step_str = filenames[__macr_names__[0]][i].split(__macr_names__[0])[-1]
        step_str = step_str[:-4]
        step = int(step_str)

        macr[step] = dict()
        for macr_name in filenames:
            macr[step][macr_name] = read_file_macr_3d(filenames[macr_name][i], path)

    return macr
