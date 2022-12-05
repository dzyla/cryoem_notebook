import datetime
import glob
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import pandas as pd
from gemmi import cif
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from plotly import graph_objects as go
from scipy import ndimage
from skimage import exposure
from skimage.transform import rescale


def save_star(dataframe_, filename='out.star', block_name='particles'):
    out_doc = cif.Document()
    out_particles = out_doc.add_new_block(block_name, pos=-1)

    # Row number is required for the column names to save the STAR file e.g. _rlnNrOfSignificantSamples #33
    column_names = dataframe_.columns
    column_names_to_star = []
    for n, name in enumerate(column_names):
        column_names_to_star.append(name + ' #{}'.format(n + 1))

    loop = out_particles.init_loop('', column_names_to_star)
    data_rows = dataframe_.to_numpy().astype(str).tolist()

    for row in data_rows:
        loop.add_row(row)

    out_doc.write_file(filename)
    # print('File "{}" saved.'.format(filename))


def save_star_31(dataframe_optics, dataframe_particles, filename='out.star', datablock_name='particles'):
    # For now only Relion star 3.1+ can be saved as 3.1 star. Adding optics will be implemented later.

    out_doc = cif.Document()
    out_particles = out_doc.add_new_block('optics', pos=-1)

    # Row number is required for the column names to save the STAR file e.g. _rlnNrOfSignificantSamples #33
    dataframe_optics = pd.DataFrame(dataframe_optics, index=[0])
    column_names = dataframe_optics.columns
    column_names_to_star = []
    for n, name in enumerate(column_names):
        column_names_to_star.append(name + ' #{}'.format(n + 1))

    loop = out_particles.init_loop('', column_names_to_star)
    data_rows = dataframe_optics.to_numpy().astype(str).tolist()

    # save optics loop
    for row in data_rows:
        loop.add_row(row)

    out_particles = out_doc.add_new_block(datablock_name, pos=-1)

    column_names = dataframe_particles.columns
    column_names_to_star = []
    for n, name in enumerate(column_names):
        column_names_to_star.append(name + ' #{}'.format(n + 1))

    loop = out_particles.init_loop('', column_names_to_star)
    data_rows = dataframe_particles.to_numpy().astype(str).tolist()

    # save particles loop
    for row in data_rows:
        loop.add_row(row)

    out_doc.write_file(filename)
    print('File "{}" saved.'.format(filename))


def convert_optics(optics_data_):
    # used for saving Relion 3.1 files with optics groups.
    # Changes the dict so values are list now.

    for key in optics_data_.keys():
        optics_data_[key] = [optics_data_[key]]

    return optics_data_


def convert_new_to_old(dataframe_, optics_group, filename, magnification='100000'):
    if optics_group == {}:
        print('File is already in Relion 3.0 format. No conversion needed!')
        quit()

    # change the Origin from Angstoms to pixels
    dataframe_['_rlnOriginXAngst'] = dataframe_['_rlnOriginXAngst'].astype(float) / optics_group[
        '_rlnImagePixelSize'].astype(float)
    dataframe_['_rlnOriginYAngst'] = dataframe_['_rlnOriginYAngst'].astype(float) / optics_group[
        '_rlnImagePixelSize'].astype(float)
    dataframe_ = dataframe_.rename(columns={'_rlnOriginXAngst': '_rlnOriginX', '_rlnOriginYAngst': '_rlnOriginY'})

    # add columns which are in the optics group
    dataframe_['_rlnVoltage'] = np.zeros(dataframe_.shape[0]) + optics_group['_rlnVoltage'].astype(float)
    dataframe_['_rlnSphericalAberration'] = np.zeros(dataframe_.shape[0]) + optics_group[
        '_rlnSphericalAberration'].astype(float)
    dataframe_['_rlnDetectorPixelSize'] = np.zeros(dataframe_.shape[0]) + optics_group['_rlnImagePixelSize'].astype(
        float)
    dataframe_['_rlnMagnification'] = np.zeros(dataframe_.shape[0]) + int(magnification)
    dataframe_['_rlnSphericalAberration'] = np.zeros(dataframe_.shape[0]) + optics_group[
        '_rlnSphericalAberration'].astype(float)

    # remove non used columns
    for tag in ['_rlnOpticsGroup', '_rlnHelicalTrackLengthAngst']:
        try:
            dataframe_ = dataframe_.drop(columns=[tag])
        except:
            pass

    # Row number is required for the column names
    column_names = dataframe_.columns
    column_names_to_star = []
    for n, name in enumerate(column_names):
        column_names_to_star.append(name + ' #{}'.format(n + 1))

    out_doc = cif.Document()
    out_particles = out_doc.add_new_block('', pos=-1)

    loop = out_particles.init_loop('', column_names_to_star)

    # to save cif all list values must be str
    data_rows = dataframe_.to_numpy().astype(str).tolist()

    for row in data_rows:
        loop.add_row(row)

    out_name = filename.replace('.star', '_v30.star')

    out_doc.write_file(out_name)
    print('File "{}" saved.'.format(out_name))


def parse_star(file_path):
    # import tqdm

    doc = cif.read_file(file_path)

    optics_data = {}

    # 3.1 star files have two data blocks Optics and particles
    _new_star_ = True if len(doc) == 2 else False

    if _new_star_:
        # print('Found Relion 3.1+ star file.')

        optics = doc[0]
        particles = doc[1]

        for item in optics:
            for optics_metadata in item.loop.tags:
                value = optics.find_loop(optics_metadata)
                optics_data[optics_metadata] = np.array(value)[0]

    else:
        # print('Found Relion 3.0 star file.')
        particles = doc[0]

    particles_data = pd.DataFrame()

    # print('Reading star file:')
    for item in particles:
        for particle_metadata in item.loop.tags:
            # If don't want to use tqdm uncomment bottom line and remove 'import tqdm'
            # for particle_metadata in item.loop.tags:
            loop = particles.find_loop(particle_metadata)
            particles_data[particle_metadata] = np.array(loop)

    return optics_data, particles_data


def parse_star_selected_columns(file_path, *kargv):
    doc = cif.read_file(file_path)

    optics_data = {}

    # 3.1 star files have two data blocks Optics and particles
    _new_star_ = True if len(doc) == 2 else False

    if _new_star_:
        # print('Found Relion 3.1+ star file.')

        optics = doc[0]
        particles = doc[1]

        for item in optics:
            for optics_metadata in item.loop.tags:
                value = optics.find_loop(optics_metadata)
                optics_data[optics_metadata] = np.array(value)[0]

    else:
        # print('Found Relion 3.0 star file.')
        particles = doc[0]

    particles_data = pd.DataFrame()

    for particle_metadata in kargv:
        loop = particles.find_loop(particle_metadata)
        particles_data[particle_metadata] = np.array(loop)

    return optics_data, particles_data


def plot_columns(particles_data, col1_name, col2_name, plot_type='hist'):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    plt.figure(figsize=(8, 8), dpi=100)

    print('Plotting {} on x and {} on y'.format(col1_name, col2_name))

    if col1_name != 'index' and col2_name != 'index':
        x_data = np.array(particles_data[col1_name].astype(float))
        y_data = np.array(particles_data[col2_name].astype(float))

    elif col1_name == 'index':
        x_data = np.arange(1, particles_data.shape[0] + 1, 1)
        y_data = np.array(particles_data[col2_name].astype(float))

    elif col2_name == 'index':
        y_data = np.arange(0, particles_data.shape[0], 1)
        x_data = np.array(particles_data[col1_name].astype(float))

    if plot_type == 'hist':
        plt.hist2d(x_data, y_data, cmap='Blues', bins=50, norm=LogNorm())
        clb = plt.colorbar()
        clb.set_label('Number of particles')

    elif plot_type == 'line':
        plt.plot(x_data, y_data)

    elif plot_type == 'scat':
        plt.scatter(x_data, y_data, cmap='Blues')

    plt.xlabel(col1_name)
    plt.ylabel(col2_name)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def plot_2dclasses_cs(path_, x_axis=None):
    # open mrcs stack
    if type(path_) == type(str()):
        classes = mrcfile.open(path_).data

    elif type(path_) == type(list()):
        classes = np.stack(path_)

    z, x, y = classes.shape
    classnumber = z
    empty_class = np.zeros((x, y))
    line = []

    # Ratio to f classes best is 5x3
    if x_axis == None:
        x_axis = int(np.sqrt(classnumber) * 1.66)
    else:
        x_axis = int(x_axis)

    if z % x_axis == 0:
        y_axis = int(z / x_axis)
    elif z % x_axis != 0:
        y_axis = int(z / x_axis) + 1

    add_extra = int(x_axis * y_axis - z)

    for n, class_ in enumerate(classes):

        if np.average(class_) != 0:
            try:
                class_ = (class_ - np.min(class_)) / (np.max(class_) - np.min(class_))
            except:
                pass

        if n == 0:
            row = class_
        else:
            if len(row) == 0:
                row = class_
            else:
                row = np.concatenate((row, class_), axis=1)
        if (n + 1) % x_axis == 0:
            line.append(row)
            row = []

    # Fill the rectangle with empty classes
    if add_extra != 0:
        for i in range(0, add_extra):
            row = np.concatenate((row, empty_class), axis=1)
        line.append(row)

    # put lines of images together so the whole rectangle is finished (as a picture)
    w = 0
    for i in line:
        if w == 0:
            final = i
            w = 1
        else:
            final = np.concatenate((final, i), axis=0)

    return final


def plot_2dclasses(path_, classnumber):
    # open mrcs stack
    if type(path_) == type(str()):
        classes = mrcfile.open(path_).data

    elif type(path_) == type(list()):
        classes = np.stack(path_)

    z, x, y = classes.shape
    empty_class = np.zeros((x, y))
    line = []

    # Ratio to display classes best is 5x3
    x_axis = int(np.sqrt(classnumber) * 1.66)

    if z % x_axis == 0:
        y_axis = int(z / x_axis)
    elif z % x_axis != 0:
        y_axis = int(z / x_axis) + 1

    add_extra = int(x_axis * y_axis - z)

    for n, class_ in enumerate(classes):

        if np.average(class_) != 0:
            try:
                class_ = (class_ - np.min(class_)) / (np.max(class_) - np.min(class_))
            except:
                pass

        if n == 0:
            row = class_
        else:
            if len(row) == 0:
                row = class_
            else:
                row = np.concatenate((row, class_), axis=1)
        if (n + 1) % x_axis == 0:
            line.append(row)
            row = []

    # Fill the rectangle with empty classes
    if add_extra != 0:
        for i in range(0, add_extra):
            row = np.concatenate((row, empty_class), axis=1)
        line.append(row)

    # put lines of images together so the whole rectangle is finished (as a picture)
    w = 0
    for i in line:
        if w == 0:
            final = i
            w = 1
        else:
            final = np.concatenate((final, i), axis=0)

    return final


def get_class_list(path_):
    class_list = []
    classes = mrcfile.open(path_).data
    for cls in classes:
        class_list.append(cls)

    return class_list


def parse_star_model(file_path, loop_name):
    doc = cif.read_file(file_path)

    # block 1 is the per class information
    loop = doc[1].find_loop(loop_name)
    class_data = np.array(loop)

    return class_data


def get_classes(path_, model_star_files):
    class_dist_per_run = []
    class_res_per_run = []

    for iter, file in enumerate(model_star_files):

        # for the refinement, plot only half2 stats
        if not 'half1' in file:
            class_dist_per_run.append(parse_star_model(file, '_rlnClassDistribution'))
            class_res_per_run.append(parse_star_model(file, '_rlnEstimatedResolution'))

    # stack all data together
    class_dist_per_run = np.stack(class_dist_per_run)
    class_res_per_run = np.stack(class_res_per_run)

    # rotate matrix so the there is class(iteration) not iteration(class) and starting from iter 0 --> iter n
    class_dist_per_run = np.flip(np.rot90(class_dist_per_run), axis=0)
    class_res_per_run = np.flip(np.rot90(class_res_per_run), axis=0)

    # Find the class images (3D) or stack (2D)
    class_files = parse_star_model(model_star_files[-1], '_rlnReferenceImage')

    class_path = []
    for class_name in class_files:
        class_name = os.path.join(path_, os.path.basename(class_name))

        # Insert only new classes, in 2D only single file
        if class_name not in class_path:
            class_path.append(class_name)

    n_classes = class_dist_per_run.shape[0]
    iter_ = class_dist_per_run.shape[1] - 1

    return class_path, n_classes, iter_, class_dist_per_run, class_res_per_run


def get_angles(path_):
    '''
    Euler angles: (rot,tilt,psi) = (φ,θ,ψ). Positive rotations of object are clockwise. Projection direction is
    defined by (rot,tilt). Psi is in-plane rotation for tilted image. For untilted rot=psi=in-plane rotation.
    Angles in a STAR file rotate the reference into observations (i.e. particle image), while translations shift
    observations into the reference projection.
    '''

    data_star = glob.glob(path_ + '/*data.star')
    data_star.sort(key=os.path.getmtime)

    last_data_star = data_star[-1]

    rot_angles = parse_star_data(last_data_star, '_rlnAngleRot').astype(float)
    tilt_angles = parse_star_data(last_data_star, '_rlnAngleTilt').astype(float)
    psi_angles = parse_star_data(last_data_star, '_rlnAnglePsi').astype(float)

    return rot_angles, tilt_angles, psi_angles


def parse_star_data(file_path, loop_name):
    do_again = True
    while do_again:
        try:
            doc = cif.read_file(file_path)

            if len(doc) == 2:
                particles_block = 1
            else:
                particles_block = 0

            # block 1 is the per class information
            loop = doc[particles_block].find_loop(loop_name)
            class_data = np.array(loop)

            do_again = False
            return class_data

        except RuntimeError:
            print('*star file is busy')
            time.sleep(5)


def plot_picks_cryosparc(cs_path, data_path, n, score, r):
    plt.figure(figsize=(10, 10), dpi=100)
    ratio = 0.2

    picking_cs = np.load(cs_path)
    mics = picking_cs['location/micrograph_path']
    unique_mics = np.unique(mics)
    mic_to_load = unique_mics[n]

    img = mrcfile.open(data_path + mic_to_load.decode("utf-8")).data
    img = rescale(img, ratio, anti_aliasing=True)

    img = mask_in_fft(img, r)
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))

    print(mic_to_load)

    try:
        data_per_mic = picking_cs[picking_cs['location/micrograph_path'] == mic_to_load]
        x_data = data_per_mic['location/center_x_frac'] * img.shape[1]
        y_data = data_per_mic['location/center_y_frac'] * img.shape[0]

        score_ncc = data_per_mic['pick_stats/ncc_score']

        norm = Normalize(vmin=score_ncc.min(), vmax=score_ncc.max())
        cmap = plt.cm.Greens
        m = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

        score_min = score_ncc.astype(float).min()
        score_max = score_ncc.astype(float).max()

        selection_low = score[0] / 100 * (score_max - score_min) + score_min
        selection_high = score[1] / 100 * (score_max - score_min)

        score_ncc = score_ncc[score_ncc <= selection_high]
        score_ncc = score_ncc[score_ncc >= selection_low]

        # score_ncc = score_ncc[score_ncc > score]

        plt.imshow(img, cmap='gray')
        plt.scatter(x_data.astype(float), y_data.astype(float),
                    s=650, facecolors='none', edgecolor=m.to_rgba(score_ncc), linewidth=1.5)

        plt.title(str('Number of picks {}\n Score min/max: {} / {}'.format(score_ncc.shape[0],
                                                                           round(selection_low, 2),
                                                                           round(selection_high, 2))))
        plt.tight_layout()
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(e)
        plt.imshow(img, cmap='gray')
        plt.tight_layout()
        plt.show()


def plot_picks_cryosparc_imported(cs_path, mic_path, n, score):
    plt.figure(figsize=(10, 10), dpi=100)
    ratio = 0.2

    picking_cs = np.load(cs_path)
    mics = picking_cs['location/micrograph_path']
    unique_mics = np.unique(mics)
    mic_to_load = unique_mics[n]
    print(os.path.basename(mic_to_load.decode("utf-8")))

    # img = mrcfile.open(mic_path + os.path.basename(mic_to_load.decode("utf-8"))).data
    img = mrcfile.open(mic_path + mic_to_load.decode("utf-8")).data
    img = rescale(img, ratio, anti_aliasing=True)

    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))

    # img = wiener(img, np.ones((5, 5)) / 2, 1)

    try:
        data_per_mic = picking_cs[picking_cs['location/micrograph_path'] == mic_to_load]
        data_per_mic = data_per_mic[data_per_mic['pick_stats/ncc_score'] > score]
        x_data = data_per_mic['location/center_x_frac'] * img.shape[0]
        y_data = data_per_mic['location/center_y_frac'] * img.shape[1]
        score_ncc = data_per_mic['pick_stats/ncc_score']
        # score_ncc = score_ncc[score_ncc > score]

        plt.imshow(img, cmap='gray')
        plt.scatter(x_data, y_data, c=score_ncc, alpha=0.4, s=100, facecolors='none')
        plt.colorbar()
        plt.title(str('Number of picks {}'.format(score_ncc.shape[0])))
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(e)
        plt.imshow(img, cmap='gray')
        plt.tight_layout()
        plt.show()


def plot_picks(img_path, n, score, picking_folder, picking_file_end):
    plt.figure(figsize=(10, 10), dpi=100)
    ratio = 0.15
    img_path = glob.glob('{}/*.mrc'.format(img_path))
    path = img_path[n]
    img = mrcfile.open(path).data
    img = rescale(img, ratio, anti_aliasing=True)
    # img = wiener(img, np.ones((5, 5)) / 2, 1)
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))
    # picking_file_end = "_autopick.star"

    try:
        picked = \
            list(parse_star('{}/{}'.format(picking_folder, os.path.basename(path).replace('.mrc', picking_file_end))))[
                1]

        if np.average(picked['_rlnAutopickFigureOfMerit'].astype(float)) != -999:
            picked = picked[picked['_rlnAutopickFigureOfMerit'].astype(float) >= score]
        else:
            score = 0
        x_data = picked['_rlnCoordinateX'].values.astype(float) * ratio
        y_data = picked['_rlnCoordinateY'].values.astype(float) * ratio
        score = picked['_rlnAutopickFigureOfMerit'].values.astype(float)
        plt.imshow(img, cmap='gray')
        plt.scatter(x_data, y_data, c=score, alpha=0.4, s=100, facecolors='none')
        plt.colorbar()
        plt.title(str('Number of picks {}'.format(score.shape[0])))
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(e)
        plt.imshow(img, cmap='gray')
        plt.tight_layout()
        plt.show()


def show_class(cls, cls_list):
    img = cls_list[cls]
    plt.imshow(img, cmap='gray')
    plt.show()


def create_circular_mask(h, w, radius=None):
    center = [int(w / 2), int(h / 2)]

    # use the smallest distance between the center and image walls
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def mask_particle(particle, radius=None):
    if radius == 1:
        return particle

    h, w = particle.shape[:2]
    radius = np.min([h, w]) / 2 * radius
    mask = create_circular_mask(h, w, radius)
    masked_img = particle.copy()
    masked_img[~mask] = 0

    return masked_img


def show_random_particles(data_folder, jobname, random_size=100, r=1, adj_contrast=False, blur=0):
    star = parse_star_whole(data_folder + 'Extract/' + jobname + '/particles.star')['particles']
    data_shape = star.shape[0]

    random_int = np.random.randint(0, data_shape, random_size)
    selected = star.iloc[random_int]

    particle_array = []
    for element in selected['_rlnImageName']:
        particle_data = element.split('@')
        img_path = data_folder + particle_data[1]
        # print(img_path)
        try:
            particle = mrcfile.mmap(img_path).data[int(particle_data[0])]
            particle = mask_in_fft(particle, r)

            if blur > 0:
                particle = ndimage.gaussian_filter(particle, blur)

            if adj_contrast:
                particle = adjust_contrast(particle)

            particle_array.append(particle)
        except IndexError:
            pass

    plt.figure(figsize=(10, 8), dpi=100)
    plt.imshow(plot_2dclasses(particle_array, len(particle_array)), cmap='gray')
    plt.axis('off')
    plt.show()


def reverse_FFT(fft_image):
    fft_img_mod = np.fft.ifftshift(fft_image)

    img_mod = np.fft.ifft2(fft_img_mod)
    return img_mod.real


def do_fft_image(img_data):
    img_fft = np.fft.fft2(img_data)

    fft_img_shift = np.fft.fftshift(img_fft)

    # real = fft_img_shift.real
    # phases = fft_img_shift.imag

    return fft_img_shift


def mask_in_fft(img, radius):
    fft = do_fft_image(img)
    masked_fft = mask_particle(fft, radius)
    img_masked = reverse_FFT(masked_fft)

    return img_masked


def show_mrc(n, r, datapath):
    plt.figure(figsize=(10, 10), dpi=100)
    img = datapath[n]
    img = mrcfile.open(img).data
    img = rescale(img, 0.2)

    img = mask_in_fft(img, r)
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))

    plt.figure(figsize=(10, 10), dpi=80)
    plt.imshow(img, cmap='gray')
    plt.show()


def plot_picks_particle(img_paths, n, particles_star):
    plt.figure(figsize=(10, 10), dpi=100)
    ratio = 0.15
    # img_path = glob.glob('{}/*.mrc'.format(img_path))
    path = img_paths[n]
    img = mrcfile.open(path).data
    img = rescale(img, ratio, anti_aliasing=True)
    # img = wiener(img, np.ones((5, 5)) / 2, 1)
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))

    try:
        optics, picking_data = parse_star_selected_columns(particles_star, '_rlnMicrographName', '_rlnCoordinateY',
                                                           '_rlnCoordinateX')
        print(picking_data.shape)

        picking_data = picking_data[picking_data[
                                        '_rlnMicrographName'] == 'MotionCorr/job069//mnt/staging/hcallaway/HMC_8-27-21_Cryo_83-122/' + os.path.basename(
            path)]

        print(picking_data.shape)

        x_data = picking_data['_rlnCoordinateX'].values.astype(float) * ratio
        y_data = picking_data['_rlnCoordinateY'].values.astype(float) * ratio
        # score = picked['_rlnAutopickFigureOfMerit'].values.astype(float)
        plt.imshow(img, cmap='gray')
        plt.scatter(x_data, y_data, alpha=0.4, s=100, facecolors='none')
        # plt.colorbar()
        plt.title(str('Number of picks {}'.format(x_data.shape[0])))
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(e)
        plt.imshow(img, cmap='gray')
        plt.tight_layout()
        plt.show()


def plot_picks_cryosparc_imported_mod(cs_path, mic_path, n, score, particles_star, show_relion):
    plt.figure(figsize=(10, 10), dpi=100)
    ratio = 0.2

    picking_cs = np.load(cs_path)
    mics = picking_cs['location/micrograph_path']
    unique_mics = np.unique(mics)
    mic_to_load = unique_mics[n]
    print(os.path.basename(mic_to_load.decode("utf-8")))

    optics, picking_data = parse_star_selected_columns(particles_star, '_rlnMicrographName', '_rlnCoordinateY',
                                                       '_rlnCoordinateX')

    micrograph_picks = picking_data[picking_data[
                                        '_rlnMicrographName'] == 'MotionCorr/job069//mnt/staging/hcallaway/HMC_8-27-21_Cryo_83-122/' + os.path.basename(
        mic_to_load.decode("utf-8")).replace('_patch_aligned_doseweighted', '')[21:]]

    # print(micrograph_picks)
    # img = mrcfile.open(mic_path + os.path.basename(mic_to_load.decode("utf-8"))).data
    img = mrcfile.open(mic_path + mic_to_load.decode("utf-8")).data
    img = rescale(img, ratio, anti_aliasing=True)

    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))

    sizex, sizey = img.shape
    # img = wiener(img, np.ones((5, 5)) / 2, 1)

    try:
        data_per_mic = picking_cs[picking_cs['location/micrograph_path'] == mic_to_load]
        data_per_mic = data_per_mic[data_per_mic['pick_stats/ncc_score'] > score]
        y_data = data_per_mic['location/center_x_frac'] * img.shape[0]
        x_data = data_per_mic['location/center_y_frac'] * img.shape[1]
        score_ncc = data_per_mic['pick_stats/ncc_score']
        # score_ncc = score_ncc[score_ncc > score]

        plt.imshow(img, cmap='gray')
        plt.scatter(x_data, y_data, c='red', alpha=0.4, s=100, facecolors='none')

        # Plot relion
        if show_relion:
            plt.scatter(micrograph_picks['_rlnCoordinateY'].astype(float) * ratio,
                        micrograph_picks['_rlnCoordinateX'].astype(float) * ratio, c='blue', alpha=0.2, s=100,
                        facecolors='none')

        plt.colorbar()
        plt.title(str('Number of picks {}'.format(score_ncc.shape[0])))
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(e)
        plt.imshow(img, cmap='gray')
        plt.tight_layout()
        plt.show()


def adjust_contrast(img, p1=2, p2=98):
    p1, p2 = np.percentile(img, (p1, p2))
    img = exposure.rescale_intensity(img, in_range=(p1, p2))
    return img


def plot_picks_new(img_path, n, score, picking_folder, picking_file_end, r, show_picks, use_img_paths=False):
    from matplotlib import cm

    plt.figure(figsize=(10, 10), dpi=100)
    ratio = 0.15
    if not use_img_paths:
        img_path = glob.glob('{}/*.mrc'.format(img_path))
    else:
        img_path = img_path

    path = img_path[n]
    img = mrcfile.open(path).data
    img = rescale(img, ratio, anti_aliasing=True)
    img = mask_in_fft(img, r)
    # img = wiener(img, np.ones((5, 5)) / 2, 1)
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))
    # picking_file_end = "_autopick.star"

    try:
        picked = \
            list(parse_star('{}/{}'.format(picking_folder, os.path.basename(path).replace('.mrc', picking_file_end))))[
                1]

        if np.average(picked['_rlnAutopickFigureOfMerit'].astype(float)) != -999:

            colormap = cm.get_cmap('plasma', len(picked))

            score_min = picked['_rlnAutopickFigureOfMerit'].astype(float).min()
            score_max = picked['_rlnAutopickFigureOfMerit'].astype(float).max()

            selection_low = score[0] * (score_max - score_min) + score_min
            selection_high = score[1] * (score_max - score_min)

            picked = picked[picked['_rlnAutopickFigureOfMerit'].astype(float) <= selection_high]
            picked = picked[picked['_rlnAutopickFigureOfMerit'].astype(float) >= selection_low]

            print(picked.shape)
        else:
            colormap = cm.get_cmap('plasma', len(picked))
            score = 0

        x_data = picked['_rlnCoordinateX'].values.astype(float) * ratio
        y_data = picked['_rlnCoordinateY'].values.astype(float) * ratio
        score = picked['_rlnAutopickFigureOfMerit'].values.astype(float)
        plt.imshow(img, cmap='gray')

        if show_picks:
            plt.scatter(x_data, y_data, s=350, facecolors='none', edgecolor=colormap.colors, linewidth=1.5)

            # plt.colorbar()
        plt.title(str('Number of picks {}\nselection values: low={}, high={}'.format(score.shape[0], selection_low,
                                                                                     selection_high)))
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(e)
        plt.imshow(img, cmap='gray')
        plt.tight_layout()
        plt.show()


def project_last_volume(path, volume_string, n_volumes):
    volume_files = glob.glob(path + "*{}.mrc".format(volume_string))
    volume_files.sort(key=os.path.getmtime)
    volume_files = volume_files[::-1]

    selected_volumes = volume_files[:n_volumes]

    print(selected_volumes)

    projection_holder = []

    for volume in selected_volumes:
        v = mrcfile.open(volume).data

        projection = np.concatenate([np.mean(v, axis=axis_) for axis_ in [0, 1, 2]], axis=1)
        p2, p98 = np.percentile(projection, (0, 99.9))
        projection = exposure.rescale_intensity(projection, in_range=(p2, p98))

        projection_holder.append(projection)

    projections = np.concatenate([arr for arr in projection_holder], axis=0)

    plt.figure(figsize=(10, 10), dpi=100)
    plt.imshow(projections, cmap='gray')
    plt.show()


def picks_from_particles(path_star, folder_particles='./Particle_picks/', data_folder=''):
    # Select the particle star file that will be split into coordinates
    particles_star = path_star

    # Saving folder name
    saving_folder = folder_particles

    # Create the folder if not present
    Path(saving_folder).mkdir(parents=True, exist_ok=True)

    # Read the star file into optics and particles sheets
    optics, data = parse_star_selected_columns(particles_star, '_rlnMicrographName', '_rlnCoordinateX',
                                               '_rlnCoordinateY')

    # Find the unique micrograph names and iterate
    for micrograph in np.unique(data['_rlnMicrographName']):
        # Select rows where micrograph name is the current one
        data = data[data['_rlnMicrographName'] == micrograph]

        # remove the column with the column name from data for saving
        data = data.drop(columns='_rlnMicrographName')

        # Get the name of the star file from the Micrograph path
        pick_star_name = os.path.basename(micrograph).replace('mrc', 'star')

        # Save the new star file per micrograph with X and Y coordinates
        save_star(data, '{}/{}'.format(saving_folder, pick_star_name), block_name='')


def cs_to_relion_from_star(cs_file, starfile_in, replace_cs, replace_rln, star_out='Rln_selected_cs.star'):
    cs_data = np.load(cs_file)
    _, star_file_imported = parse_star(starfile_in)

    selected_bound_df = pd.DataFrame()
    selected_bound_df['blob/path'] = cs_data['blob/path']
    selected_bound_df['blob/idx'] = cs_data['blob/idx']
    selected_bound_df['blob/path'] = selected_bound_df['blob/path'].str.decode("utf-8").str.replace(replace_cs,
                                                                                                    replace_rln)

    selected_bound_df['blob/idx'] = selected_bound_df['blob/idx'].astype(str).str.zfill(6)

    selected_bound_df['_rlnImageName'] = selected_bound_df['blob/idx'].astype(str) + selected_bound_df['blob/path']

    selected_names = selected_bound_df['_rlnImageName']

    selected_from_cs = star_file_imported[star_file_imported['_rlnImageName'].isin(selected_names)]
    save_star_31(_, selected_from_cs, filename=star_out)


def plot_ctf_stats(starctf, index, defocus, max_res, fom, save_star=False, optics={}, data_folder=''):
    from matplotlib.colors import LogNorm
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    reduced_ctfcorrected = starctf[0]

    index_low, index_high = np.percentile(range(reduced_ctfcorrected.shape[0]), (index[0], index[1]))

    defocus_low, defocus_high = np.percentile(np.linspace(reduced_ctfcorrected['_rlnDefocusV'].astype(float).min(),
                                                          reduced_ctfcorrected['_rlnDefocusV'].astype(float).max()),
                                              (defocus[0], defocus[1]))

    max_res_low, max_res_high = np.percentile(
        np.linspace(reduced_ctfcorrected['_rlnCtfMaxResolution'].astype(float).min(),
                    reduced_ctfcorrected['_rlnCtfMaxResolution'].astype(float).max()),
        (max_res[0], max_res[1]))
    fom_low, fom_high = np.percentile(np.linspace(reduced_ctfcorrected['_rlnCtfFigureOfMerit'].astype(float).min(),
                                                  reduced_ctfcorrected['_rlnCtfFigureOfMerit'].astype(float).max()),
                                      (fom[0], fom[1]))

    reduced_ctfcorrected = reduced_ctfcorrected[int(index_low):int(index_high)]

    # selection
    reduced_ctfcorrected = reduced_ctfcorrected[(defocus_low <= reduced_ctfcorrected['_rlnDefocusV'].astype(float)) &
                                                (reduced_ctfcorrected['_rlnDefocusV'].astype(float) <= defocus_high)]

    reduced_ctfcorrected = reduced_ctfcorrected[
        (max_res_low <= reduced_ctfcorrected['_rlnCtfMaxResolution'].astype(float)) &
        (reduced_ctfcorrected['_rlnCtfMaxResolution'].astype(float) <= max_res_high)]

    reduced_ctfcorrected = reduced_ctfcorrected[
        (fom_low <= reduced_ctfcorrected['_rlnCtfFigureOfMerit'].astype(float)) &
        (reduced_ctfcorrected['_rlnCtfFigureOfMerit'].astype(float) <= fom_high)]

    print('Selected files: {}, Selected defocus: {}-{}, Selected Ctf resolution: {}-{}, Selected FOM: {}-{}'.format(
        reduced_ctfcorrected.shape, defocus_low, defocus_high, max_res_low, max_res_high, fom_low, fom_high))

    axs[0, 0].plot(range(0, reduced_ctfcorrected.shape[0]),
                   reduced_ctfcorrected['_rlnDefocusV'].astype(float))
    axs[0, 0].set_xlabel('Index')
    axs[0, 0].set_ylabel('_rlnDefocusV')

    axs[0, 1].plot(range(0, reduced_ctfcorrected.shape[0]),
                   reduced_ctfcorrected['_rlnCtfMaxResolution'].astype(float))
    axs[0, 1].set_xlabel('Index')
    axs[0, 1].set_ylabel('_rlnCtfMaxResolution')

    axs[1, 0].hist2d(reduced_ctfcorrected['_rlnDefocusV'].astype(float),
                     reduced_ctfcorrected['_rlnCtfMaxResolution'].astype(float),
                     bins=50, norm=LogNorm(), cmap='Blues')
    axs[1, 0].set_xlabel('_rlnDefocusV')
    axs[1, 0].set_ylabel('_rlnCtfMaxResolution')

    axs[1, 1].hist2d(reduced_ctfcorrected['_rlnDefocusV'].astype(float),
                     reduced_ctfcorrected['_rlnCtfFigureOfMerit'].astype(float),
                     bins=50, norm=LogNorm(), cmap='Blues')
    axs[1, 1].set_xlabel('_rlnDefocusV')
    axs[1, 1].set_ylabel('_rlnCtfFigureOfMerit')
    plt.show()

    if save_star:
        save_star_31(optics, reduced_ctfcorrected, '{}/micrographs_ctf_selected.star'.format(data_folder))
        save_star = False

    return reduced_ctfcorrected


class class3d_run:
    def __init__(self, data_folder, job_n, n_cls):
        self.folder = data_folder
        self.path = self.folder + '/Class3D/{}/'.format(job_n)
        self.angles()
        self.cls_stats()
        self.n_cls = n_cls
        # self.run_params()

    def angles(self):
        self.rot, self.tilt, self.psi = get_angles(self.path)

    def cls_stats(self):
        model_star = glob.glob(self.path + '/*model.star')
        model_star.sort(key=os.path.getmtime)
        self.model_star = model_star
        self.class_path, self.n_classes, self.iter_, self.class_dist_per_run, self.class_res_per_run = get_classes(
            self.path, self.model_star)

    def plot_stats_dist(self):
        self.cls_stats()
        plt.figure(figsize=(8, 4), dpi=100)

        for N, cls_ in enumerate(self.class_dist_per_run):
            plt.plot(range(0, len(cls_)), cls_.astype(float), label='{}'.format(N))

        plt.xlabel('Iter')
        plt.ylabel('Class distribution')
        plt.legend()

        plt.show()

    def plot_stats_res(self):
        self.cls_stats()
        plt.figure(figsize=(8, 4), dpi=100)

        for N, cls_ in enumerate(self.class_res_per_run):
            plt.plot(range(0, len(cls_)), cls_.astype(float), label='{}'.format(N))

        plt.xlabel('Iter')
        plt.ylabel('Resolution, A')
        plt.legend()
        plt.show()

    def plot_all_classes(self):
        self.cls_stats()
        project_last_volume(self.path, '', self.n_cls)

    def run_params(self):
        with open(self.path + 'note.txt') as file:
            print(file.read())


def parse_star_whole(file_path):
    doc = cif.read_file(file_path)

    star_data = {}

    for data in doc:
        try:

            dataframe = pd.DataFrame()
            for item in data:

                for metadata in item.loop.tags:
                    value = data.find_loop(metadata)
                    dataframe[metadata] = np.array(value)
                star_data[data.name] = dataframe

        except AttributeError as e:
            pass
            # print(e)

    return star_data


def display_2dclasses(
        images, model_star=[],
        columns=10, width=20, height=2,
        label_wrap_length=10, label_font_size=8, sort=False, label=True):
    import textwrap

    if model_star:
        labels = []
        cls_dist = []
        for n, dist in enumerate(parse_star_whole(model_star)['model_classes']['_rlnClassDistribution']):
            labels.append('Class {} {}%'.format(n + 1, round(float(dist) * 100, 2)))
            cls_dist.append(float(dist))

    max_images = len(images)

    if sort and model_star:
        sort_matrix = np.argsort(cls_dist)[::-1]
        images = np.array(images)[sort_matrix]
        labels = np.array(labels)[sort_matrix]

    height = max(height, int(len(images) / columns) * height)
    plt.figure(figsize=(width, height))
    for i, image in enumerate(images):

        plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        plt.imshow(image, cmap='gray')

        if label:
            if model_star:
                title = labels[i]
                title = textwrap.wrap(title, label_wrap_length)
                title = "\n".join(title)
                plt.title(title, fontsize=label_font_size);
        plt.axis('off')
        plt.tight_layout()


def rot2euler(r):
    """Decompose rotation matrix into Euler angles"""
    # assert(isrotation(r))
    # Shoemake rotation matrix decomposition algorithm with same conventions as Relion.
    epsilon = np.finfo(np.double).eps
    abs_sb = np.sqrt(r[0, 2] ** 2 + r[1, 2] ** 2)
    if abs_sb > 16 * epsilon:
        gamma = np.arctan2(r[1, 2], -r[0, 2])
        alpha = np.arctan2(r[2, 1], r[2, 0])
        if np.abs(np.sin(gamma)) < epsilon:
            sign_sb = np.sign(-r[0, 2]) / np.cos(gamma)
        else:
            sign_sb = np.sign(r[1, 2]) if np.sin(gamma) > 0 else -np.sign(r[1, 2])
        beta = np.arctan2(sign_sb * abs_sb, r[2, 2])
    else:
        if np.sign(r[2, 2]) > 0:
            alpha = 0
            beta = 0
            gamma = np.arctan2(-r[1, 0], r[0, 0])
        else:
            alpha = 0
            beta = np.pi
            gamma = np.arctan2(r[1, 0], -r[0, 0])
    return alpha, beta, gamma


def expmap(e):
    """Convert axis-angle vector into 3D rotation matrix"""
    theta = np.linalg.norm(e)
    if theta < 1e-16:
        return np.identity(3, dtype=e.dtype)
    w = e / theta
    k = np.array([[0, w[2], -w[1]],
                  [-w[2], 0, w[0]],
                  [w[1], -w[0], 0]], dtype=e.dtype)
    r = np.identity(3, dtype=e.dtype) + np.sin(theta) * k + (1 - np.cos(theta)) * np.dot(k, k)
    return r


def pose_to_angles(p0, p1, p2):
    rot_matrix = np.array([p0, p1, p2])
    ang = np.array([np.rad2deg(rot2euler(expmap(x))) for x in rot_matrix])
    return ang


def cs_to_pd(file_path):
    df = pd.DataFrame()
    counter = 0

    data = np.load(file_path)
    for name in data.dtype.names:
        try:
            df[name] = data[name]

        except Exception as e:
            for column in range(data[name].shape[1]):
                df[name + '_{}'.format(counter)] = data[name][:, column]
                counter += 1
            counter = 0

    return df


def display_2dclasses_cs(cs_folder, cls2d_cs_data,
                         columns=10, width=20, height=2,
                         label_wrap_length=10, label_font_size=8, sort=False, percent=False):
    import textwrap, os

    class_averages_files = glob.glob(cs_folder + '/*class_averages.mrc')
    class_averages_files.sort(key=os.path.getmtime)

    images = mrcfile.open(class_averages_files[-1]).data

    particles_cs = glob.glob(cs_folder + '/*_particles.cs')
    particles_cs = [file for file in particles_cs if 'passthrough' not in file]

    particles_cs.sort(key=os.path.getmtime)

    particles_cs = particles_cs[-1]

    labels = []

    cs_data = cs_to_pd(cls2d_cs_data)

    cls_dist = np.unique(cs_data['alignments2D/class'], return_counts=True)[1]

    if percent:
        cls_dist = cls_dist / np.sum(cls_dist)

        for n, dist in enumerate(cls_dist):
            labels.append('Class {} {}%'.format(n, round(float(dist) * 100, 2)))

    else:
        for n, dist in enumerate(cls_dist):
            labels.append('Class {} {} ptcls.'.format(n, dist))

    max_images = len(images)

    if sort and particles_cs:
        sort_matrix = np.argsort(cls_dist)[::-1]
        images = np.array(images)[sort_matrix]
        labels = np.array(labels)[sort_matrix]

    height = max(height, int(len(images) / columns) * height)
    plt.figure(figsize=(width, height))
    for i, image in enumerate(images):

        plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        plt.imshow(image, cmap='gray')

        if particles_cs:
            title = labels[i]
            title = textwrap.wrap(title, label_wrap_length)
            title = "\n".join(title)
            plt.title(title, fontsize=label_font_size);
        plt.axis('off')
        plt.tight_layout()


def get_cylinder(file, cylinder_radius=None, reverse=False):
    lx, ly, lz = file.shape
    X, Y = np.ogrid[0:lx, 0:ly]
    dist_from_center = np.sqrt((X - lx / 2) ** 2 + (Y - ly / 2) ** 2)

    if not reverse:
        if cylinder_radius != None:
            mask = dist_from_center <= cylinder_radius
        else:
            mask = dist_from_center <= lx / 2
    else:
        if cylinder_radius != None:
            mask = dist_from_center >= cylinder_radius
        else:
            mask = dist_from_center >= lx / 2

    new = np.zeros(file.shape)
    for n in range(0, new.shape[0]):
        new[n] = mask
    return new


def squarify(M, val):
    (a, b) = M.shape
    if a > b:
        padding = ((0, 0), (0, a - b))
    else:
        padding = ((0, b - a), (0, 0))
    return np.pad(M, padding, mode='constant', constant_values=val)


def show_mrc_fft(n, r, datapath, fft=False, blur=False, filter_intensity=5):
    plt.figure(figsize=(10, 10), dpi=150)
    img = datapath[n]
    img = mrcfile.open(img).data.astype(float)  # [300:1200,300:1200]

    if not fft:
        img = rescale(img, 0.2)
        img = mask_in_fft(img, r)
        p2, p98 = np.percentile(img, (2, 98))
    else:
        if img.shape[0] != img.shape[1]:
            img = squarify(img, 0)
        img = np.log(do_fft_image(img).real ** 2)
        p2, p98 = np.percentile(img, (2, 100))
        img = rescale(img, 0.2)

    if blur:
        img = ndimage.gaussian_filter(img, filter_intensity)
    img = exposure.rescale_intensity(img, in_range=(p2, p98))

    plt.figure(figsize=(10, 10), dpi=80)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def plot_ctf_stats(starctf, index, defocus, max_res, fom, save_star=False, data_folder=''):
    from matplotlib.colors import LogNorm
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), dpi=300)

    reduced_ctfcorrected = starctf['micrographs']
    optics = starctf['optics']

    index_low, index_high = np.percentile(range(reduced_ctfcorrected.shape[0]), (index[0], index[1]))

    defocus_low, defocus_high = np.percentile(np.linspace(reduced_ctfcorrected['_rlnDefocusV'].astype(float).min(),
                                                          reduced_ctfcorrected['_rlnDefocusV'].astype(float).max()),
                                              (defocus[0], defocus[1]))

    max_res_low, max_res_high = np.percentile(
        np.linspace(reduced_ctfcorrected['_rlnCtfMaxResolution'].astype(float).min(),
                    reduced_ctfcorrected['_rlnCtfMaxResolution'].astype(float).max()),
        (max_res[0], max_res[1]))
    fom_low, fom_high = np.percentile(np.linspace(reduced_ctfcorrected['_rlnCtfFigureOfMerit'].astype(float).min(),
                                                  reduced_ctfcorrected['_rlnCtfFigureOfMerit'].astype(float).max()),
                                      (fom[0], fom[1]))

    reduced_ctfcorrected = reduced_ctfcorrected[int(index_low):int(index_high)]

    # selection
    reduced_ctfcorrected = reduced_ctfcorrected[(defocus_low <= reduced_ctfcorrected['_rlnDefocusV'].astype(float)) &
                                                (reduced_ctfcorrected['_rlnDefocusV'].astype(float) <= defocus_high)]

    reduced_ctfcorrected = reduced_ctfcorrected[
        (max_res_low <= reduced_ctfcorrected['_rlnCtfMaxResolution'].astype(float)) &
        (reduced_ctfcorrected['_rlnCtfMaxResolution'].astype(float) <= max_res_high)]

    reduced_ctfcorrected = reduced_ctfcorrected[
        (fom_low <= reduced_ctfcorrected['_rlnCtfFigureOfMerit'].astype(float)) &
        (reduced_ctfcorrected['_rlnCtfFigureOfMerit'].astype(float) <= fom_high)]

    print('Selected files: {}, Selected defocus: {}-{}, Selected Ctf resolution: {}-{}, Selected FOM: {}-{}'.format(
        reduced_ctfcorrected.shape[0], round(defocus_low, 1), round(defocus_high, 1),
        round(max_res_low, 1), round(max_res_high, 1), round(fom_low, 2), round(fom_high, 2)))

    axs[0, 0].plot(range(0, reduced_ctfcorrected.shape[0]),
                   reduced_ctfcorrected['_rlnDefocusV'].astype(float))
    axs[0, 0].set_xlabel('Index')
    axs[0, 0].set_ylabel('_rlnDefocusV')

    axs[0, 1].plot(range(0, reduced_ctfcorrected.shape[0]),
                   reduced_ctfcorrected['_rlnCtfMaxResolution'].astype(float))
    axs[0, 1].set_xlabel('Index')
    axs[0, 1].set_ylabel('_rlnCtfMaxResolution')

    axs[1, 0].hist2d(reduced_ctfcorrected['_rlnDefocusV'].astype(float),
                     reduced_ctfcorrected['_rlnCtfMaxResolution'].astype(float),
                     bins=50, norm=LogNorm(), cmap='Blues')
    axs[1, 0].set_xlabel('_rlnDefocusV')
    axs[1, 0].set_ylabel('_rlnCtfMaxResolution')

    axs[1, 1].hist2d(reduced_ctfcorrected['_rlnDefocusV'].astype(float),
                     reduced_ctfcorrected['_rlnCtfFigureOfMerit'].astype(float),
                     bins=50, norm=LogNorm(), cmap='Blues')
    axs[1, 1].set_xlabel('_rlnDefocusV')
    axs[1, 1].set_ylabel('_rlnCtfFigureOfMerit')
    plt.show()

    if save_star:
        save_star_31(optics, reduced_ctfcorrected, '{}/micrographs_ctf_selected.star'.format(data_folder),
                     'micrographs')
        save_star = False

    return reduced_ctfcorrected


def save_star_31(dataframe_optics, dataframe_particles, filename='out.star', datablock_name='particles'):
    # For now only Relion star 3.1+ can be saved as 3.1 star. Adding optics will be implemented later.

    out_doc = cif.Document()
    out_particles = out_doc.add_new_block('optics', pos=-1)

    # Row number is required for the column names to save the STAR file e.g. _rlnNrOfSignificantSamples #33
    dataframe_optics = pd.DataFrame(dataframe_optics, index=[0])
    column_names = dataframe_optics.columns
    column_names_to_star = []
    for n, name in enumerate(column_names):
        column_names_to_star.append(name + ' #{}'.format(n + 1))

    loop = out_particles.init_loop('', column_names_to_star)
    data_rows = dataframe_optics.to_numpy().astype(str).tolist()

    # save optics loop
    for row in data_rows:
        loop.add_row(row)

    out_particles = out_doc.add_new_block(datablock_name, pos=-1)

    column_names = dataframe_particles.columns
    column_names_to_star = []
    for n, name in enumerate(column_names):
        column_names_to_star.append(name + ' #{}'.format(n + 1))

    loop = out_particles.init_loop('', column_names_to_star)
    data_rows = dataframe_particles.to_numpy().astype(str).tolist()

    # save particles loop
    for row in data_rows:
        loop.add_row(row)

    out_doc.write_file(filename)
    print('File "{}" saved.'.format(filename))


def load_topaz_curve(file, pyplot=False):
    topaz_training_txt = file

    data = pd.read_csv(topaz_training_txt, delimiter='\t')
    data_test = data[data['split'] == 'test']

    x = data_test['epoch']
    data_test = data_test.drop(['iter', 'split', 'ge_penalty'], axis=1)

    if not pyplot:
        fig_ = go.Figure()

        for n, column in enumerate(data_test.columns):
            if column != 'epoch':
                y = data_test[column]
                fig_.add_scatter(x=x, y=y, name='{}'.format(column))

        fig_.update_xaxes(title_text="Epoch")
        fig_.update_yaxes(title_text="Statistics")

        fig_.update_layout(
            title="Topaz training stats. Best model: {}".format(data_test[data_test['auprc'].astype(float) ==
                                                                          np.max(data_test['auprc'].astype(float))][
                                                                    'epoch'].values)
        )

        fig_.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        fig_.show()
    else:
        for n, column in enumerate(data_test.columns):
            if column != 'epoch':
                y = data_test[column]
                plt.plot(x.astype(float), y.astype(float), label='{}'.format(column), marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Statistics")
        plt.title("Topaz training stats. Best model: {}".format(data_test[data_test['auprc'].astype(float) ==
                                                                          np.max(data_test['auprc'].astype(float))][
                                                                    'epoch'].values))
        plt.legend()
        plt.show()


class class3d_run:
    def __init__(self, data_folder, job_n, n_cls, dpi=200, max_project=False, bins=20):
        self.folder = data_folder
        self.path = self.folder + '/Class3D/{}/'.format(job_n)
        self.angles()
        self.cls_stats()
        self.n_cls = n_cls
        self.dpi = dpi
        self.max_project = max_project
        self.bins = bins
        # self.run_params()

    def angles(self):
        self.rot, self.tilt, self.psi = get_angles(self.path)

    def cls_stats(self):
        model_star = glob.glob(self.path + '/*model.star')
        model_star.sort(key=os.path.getmtime)
        self.model_star = model_star
        self.class_path, self.n_classes, self.iter_, self.class_dist_per_run, self.class_res_per_run = get_classes(
            self.path, self.model_star)

    def plot_stats_dist(self):
        self.cls_stats()
        plt.figure(figsize=(8, 4), dpi=100)

        for N, cls_ in enumerate(self.class_dist_per_run):
            plt.plot(range(0, len(cls_)), cls_.astype(float), label='{}'.format(N))

        plt.xlabel('Iter')
        plt.ylabel('Class distribution')
        plt.legend()

        plt.show()

    def plot_stats_res(self):
        self.cls_stats()
        plt.figure(figsize=(8, 4), dpi=100)

        for N, cls_ in enumerate(self.class_res_per_run):
            plt.plot(range(0, len(cls_)), cls_.astype(float), label='{}'.format(N))

        plt.xlabel('Iter')
        plt.ylabel('Resolution, A')
        plt.legend()
        plt.show()

    def plot_all_classes(self):
        self.cls_stats()

        model_files = glob.glob(self.path + "/*model.star")
        model_files.sort(key=os.path.getmtime)

        n_inter = len(model_files)

        (class_paths, n_classes_, iter_, class_dist_, class_res_) = get_classes(self.path, model_files)

        cls3d_projections = plot_3dclasses(class_paths, max_projection=self.max_project, joining_axes=[1, 0])

        shp_x = cls3d_projections.shape[1] / 3
        shp_y = cls3d_projections.shape[0]

        plt.figure(dpi=self.dpi)
        plt.imshow(cls3d_projections, cmap='gray')
        plt.yticks([0.5 * shp_y], ['projection'], rotation=90)
        plt.xticks([0.5 * shp_x, 0.5 * shp_x + shp_x, 0.5 * shp_x + shp_x * 2], ['z', 'x', 'y'])

        plt.show()

        # project_last_volume(self.path, '', self.n_cls)

    def run_params(self):
        with open(self.path + 'note.txt') as file:
            print(file.read())

    def plot_angles(self):
        self.angles()
        plt.figure(figsize=(8, 4), dpi=100)

        plt.hist2d(self.psi, self.rot, bins=self.bins)
        plt.xlabel('Rot')
        plt.ylabel('Tilt')
        plt.show()

class refine3d_run:
    def __init__(self, data_folder, job_n, n_cls=2, dpi=200, max_project=False, bins=20):
        self.folder = data_folder
        self.path = self.folder + '/Refine3D/{}/'.format(job_n)
        self.angles()
        self.cls_stats()
        self.n_cls = n_cls
        self.dpi = dpi
        self.max_project = max_project
        self.bins = bins
        # self.run_params()

    def angles(self):
        self.rot, self.tilt, self.psi = get_angles(self.path)

    def cls_stats(self):
        model_star = glob.glob(self.path + '/*model.star')
        model_star.sort(key=os.path.getmtime)
        self.model_star = model_star
        self.class_path, self.n_classes, self.iter_, self.class_dist_per_run, self.class_res_per_run = get_classes(
            self.path, self.model_star)

    def plot_stats_dist(self):
        self.cls_stats()
        plt.figure(figsize=(8, 4), dpi=100)

        for N, cls_ in enumerate(self.class_dist_per_run):
            plt.plot(range(0, len(cls_)), cls_.astype(float), label='{}'.format(N))

        plt.xlabel('Iter')
        plt.ylabel('Class distribution')
        plt.legend()

        plt.show()

    def plot_stats_res(self):
        self.cls_stats()
        plt.figure(figsize=(8, 4), dpi=100)

        for N, cls_ in enumerate(self.class_res_per_run):
            plt.plot(range(0, len(cls_)), cls_.astype(float), label='{}'.format(N))

        plt.xlabel('Iter')
        plt.ylabel('Resolution, A')
        plt.legend()
        plt.show()

    def plot_all_classes(self):
        self.cls_stats()

        model_files = glob.glob(self.path + "/*model.star")
        model_files.sort(key=os.path.getmtime)

        n_inter = len(model_files)

        (class_paths, n_classes_, iter_, class_dist_, class_res_) = get_classes(self.path, model_files)

        cls3d_projections = plot_3dclasses(class_paths, max_projection=self.max_project, joining_axes=[1, 0])

        shp_x = cls3d_projections.shape[1] / 3
        shp_y = cls3d_projections.shape[0]

        plt.figure(dpi=self.dpi)
        plt.imshow(cls3d_projections, cmap='gray')
        plt.yticks([0.5 * shp_y], ['projection'], rotation=90)
        plt.xticks([0.5 * shp_x, 0.5 * shp_x + shp_x, 0.5 * shp_x + shp_x * 2], ['z', 'x', 'y'])

        plt.show()

        # project_last_volume(self.path, '', self.n_cls)

    def run_params(self):
        with open(self.path + 'note.txt') as file:
            print(file.read())

    def plot_angles(self):
        self.angles()
        plt.figure(figsize=(8, 4), dpi=100)

        plt.hist2d(self.psi, self.rot, bins=self.bins)
        plt.xlabel('Rot')
        plt.ylabel('Tilt')
        plt.show()

def plot_picks_relion(rln_folder, job_name, n, score, r, blur=False, filter_intensity=5):
    import matplotlib as mpl
    from matplotlib import cm

    autopick_shorcode = []
    img_resize_fac = 0.2
    path_data = rln_folder + job_name

    coordinate_files = glob.glob(path_data + "*.star")
    coordinate_files.sort(key=os.path.getmtime)

    # Relion 4 has much easier coordinate handling
    if os.path.exists(path_data + 'autopick.star') and os.path.getsize(path_data + 'autopick.star') > 0:
        pick_old = False

        autopick_star = parse_star_whole(path_data + 'autopick.star')['coordinate_files']
        mics_paths = autopick_star['_rlnMicrographName']
        coord_paths = autopick_star['_rlnMicrographCoordinates']
        if coord_paths.shape[0] != 1:
            coord_paths = np.squeeze(coord_paths.to_numpy())

    elif os.path.exists(path_data + 'manualpick.star') and os.path.getsize(path_data + 'manualpick.star') > 0:
        pick_old = False

        manpick_star = parse_star_whole(path_data + 'manualpick.star')['coordinate_files']
        mics_paths = manpick_star['_rlnMicrographName']
        coord_paths = manpick_star['_rlnMicrographCoordinates']
        if coord_paths.shape[0] != 1:
            coord_paths = np.squeeze(coord_paths.to_numpy())

    elif glob.glob(path_data + 'coords_suffix_*') != []:
        # Get the coordinates from subfolders
        pick_old = True

        # get suffix firsts

        suffix_file = glob.glob(path_data + 'coords_suffix_*')[0]
        suffix = os.path.basename(suffix_file).replace('coords_suffix_', '').replace('.star', '')

        # Get the folder with micrographs
        mics_data_path = open(glob.glob(path_data + 'coords_suffix_*')[0]).readlines()[0].replace('\n', '')

        all_mics_paths = parse_star_data(rln_folder + mics_data_path, '_rlnMicrographName')

        mics_paths = []
        for name in all_mics_paths:
            mics_paths.append(rln_folder + name)

    try:
        file = np.array(mics_paths)[n]

        if pick_old:

            # Create a picking star file name based on the mic name
            pick_star_name = os.path.basename(file).replace('.mrc', '_{}.star'.format(suffix))

            # Find the file location if there
            pick_star_name_path = glob.glob(path_data + '/**/' + pick_star_name)

            # if the file is there go, if not it will be empty list
            if pick_star_name_path != []:
                # Open micrograph and corresponding star file
                micrograph = mrcfile.open(file, permissive=True).data
                coords_file = parse_star(pick_star_name_path[0])[1]
                coords_x = coords_file['_rlnCoordinateX']
                coords_y = coords_file['_rlnCoordinateY']
                autopick_fom = coords_file['_rlnAutopickFigureOfMerit']


        # If relion 4 and has nicely placed all files together in autopick.star
        else:

            micrograph = mrcfile.open(rln_folder + file, permissive=True).data
            coords_file = parse_star(rln_folder + coord_paths[n])[1]
            coords_x = coords_file['_rlnCoordinateX']
            coords_y = coords_file['_rlnCoordinateY']
            autopick_fom = coords_file['_rlnAutopickFigureOfMerit']

        # Make plots
        mic_red = rescale(micrograph.astype(float), img_resize_fac)
        mic_red = mask_in_fft(mic_red, r)

        # plt.hist(mic_red.flatten(), bins=100)
        # plt.show()

        p1, p2 = np.percentile(mic_red, (0.1, 99.8))
        mic_red = np.array(exposure.rescale_intensity(mic_red, in_range=(p1, p2)))

        if blur:
            mic_red = ndimage.gaussian_filter(mic_red, filter_intensity)

        try:
            norm = mpl.colors.Normalize(vmin=autopick_fom.astype(float).min(), vmax=autopick_fom.astype(float).max())
            cmap = cm.Greens
            m = cm.ScalarMappable(norm=norm, cmap=cmap)

            score_min = autopick_fom.astype(float).min()
            score_max = autopick_fom.astype(float).max()

            selection_low = score[0] / 100 * (score_max - score_min) + score_min
            selection_high = score[1] / 100 * (score_max - score_min)

            coords_x = coords_x[autopick_fom.astype(float) <= selection_high]
            coords_y = coords_y[autopick_fom.astype(float) <= selection_high]
            autopick_fom = autopick_fom[autopick_fom.astype(float) <= selection_high]

            coords_x = coords_x[autopick_fom.astype(float) >= selection_low]
            coords_y = coords_y[autopick_fom.astype(float) >= selection_low]
            autopick_fom = autopick_fom[autopick_fom.astype(float) >= selection_low]

            plt.figure(figsize=(10, 10), dpi=100)
            plt.imshow(mic_red, cmap='gray')
            plt.axis('off')

            plt.scatter(coords_x.astype(float) * img_resize_fac, coords_y.astype(float) * img_resize_fac,
                        s=750, facecolors='none', edgecolor=m.to_rgba(autopick_fom.astype(float)), linewidth=1.2)

            plt.title(str('Number of picks {}\n Score min/max: {} / {}'.format(autopick_fom.shape[0],
                                                                               round(selection_low, 2),
                                                                               round(selection_high, 2))))
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(e)

            plt.scatter(coords_x.astype(float) * img_resize_fac, coords_y.astype(float) * img_resize_fac,
                        s=750, facecolors='none', edgecolor='green', linewidth=1.2)

            plt.title(str('Number of picks {}'.format(coords_x.shape[0])))
            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(e)


def get_angles(path_):
    '''
    Euler angles: (rot,tilt,psi) = (φ,θ,ψ). Positive rotations of object are clockwise. Projection direction is
    defined by (rot,tilt). Psi is in-plane rotation for tilted image. For untilted rot=psi=in-plane rotation.
    Angles in a STAR file rotate the reference into observations (i.e. particle image), while translations shift
    observations into the reference projection.
    '''

    data_star = glob.glob(path_ + '/*data.star')
    data_star.sort(key=os.path.getmtime)

    last_data_star = data_star[-1]

    rot_angles = parse_star_data(last_data_star, '_rlnAngleRot').astype(float)
    tilt_angles = parse_star_data(last_data_star, '_rlnAngleTilt').astype(float)
    psi_angles = parse_star_data(last_data_star, '_rlnAnglePsi').astype(float)

    return rot_angles, tilt_angles, psi_angles


def display_2dclasses_cs(cs_folder,
                         columns=10, width=20, height=2,
                         label_wrap_length=10, label_font_size=8, sort=False, percent=False, label=True):
    import textwrap, os

    class_averages_files = glob.glob(cs_folder + '/*class_averages.mrc')
    class_averages_files.sort(key=os.path.getmtime)

    images = mrcfile.open(class_averages_files[-1]).data

    particles_cs = glob.glob(cs_folder + '/*_particles.cs')
    particles_cs = [file for file in particles_cs if 'passthrough' not in file]

    particles_cs.sort(key=os.path.getmtime)

    particles_cs = particles_cs[-1]

    labels = []

    cs_data = cs_to_pd(particles_cs)

    cls_dist = np.unique(cs_data['alignments2D/class'], return_counts=True)[1]

    if percent:
        cls_dist = cls_dist / np.sum(cls_dist)

        for n, dist in enumerate(cls_dist):
            labels.append('Class {} {}%'.format(n, round(float(dist) * 100, 2)))

    else:
        for n, dist in enumerate(cls_dist):
            labels.append('Class {} {} ptcls.'.format(n, dist))

    max_images = len(images)

    if sort:
        sort_matrix = np.argsort(cls_dist)[::-1]
        images = np.array(images)[sort_matrix]
        labels = np.array(labels)[sort_matrix]

    height = max(height, int(len(images) / columns) * height)
    plt.figure(figsize=(width, height))
    for i, image in enumerate(images):

        plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        plt.imshow(image, cmap='gray')

        if particles_cs:
            if label:
                title = labels[i]
                title = textwrap.wrap(title, label_wrap_length)
                title = "\n".join(title)
                plt.title(title, fontsize=label_font_size);
        plt.axis('off')
        plt.tight_layout()


class class2d_run_plotly:
    def __init__(self, data_folder, job_n):
        self.folder = data_folder
        self.path = self.folder + '/Class2D/{}/'.format(job_n)
        self.angles()
        self.cls_stats()
        # self.run_params()

    def angles(self):
        self.rot, self.tilt, self.psi = get_angles(self.path)

    def cls_stats(self):
        model_star = glob.glob(self.path + '/*model.star')
        model_star.sort(key=os.path.getmtime)
        self.model_star = model_star
        self.class_path, self.n_classes, self.iter_, self.class_dist_per_run, self.class_res_per_run = get_classes(
            self.path, self.model_star)

    def plot_dist(self):
        self.cls_stats()

        fig_ = go.Figure()

        for n, class_ in enumerate(self.class_dist_per_run):
            class_ = np.float16(class_)
            x = np.arange(0, self.class_dist_per_run.shape[1])

            fig_.add_scatter(x=x, y=class_, name='class {}'.format(n + 1), showlegend=False)

        fig_.update_xaxes(title_text="Iteration")
        fig_.update_yaxes(title_text="Class distribution")

        fig_.update_layout(
            title="Class distribution"
        )

        fig_.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        fig_.show()

    def plot_res(self):
        self.cls_stats()
        fig_ = go.Figure()

        for n, class_ in enumerate(self.class_res_per_run):
            class_ = np.float16(class_)
            x = np.arange(0, self.class_res_per_run.shape[1])

            fig_.add_scatter(x=x, y=class_, name='class {}'.format(n + 1), showlegend=False)

        fig_.update_xaxes(title_text="Iteration")
        fig_.update_yaxes(title_text="Class Resolution [A]")

        fig_.update_layout(
            title="Class Resolution [A]"
        )

        fig_.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        fig_.show()

    def plot_classes(self, label=True, columns_=10, width_=15):
        self.cls_stats()

        display_2dclasses(images=self.get_2dcls(), model_star=self.model_star[-1], sort=True, columns=columns_,
                          width=width_, label=label)

    def get_2dcls(self):
        self.cls_stats()
        self.cls_list = get_class_list(self.class_path[-1])
        return self.cls_list

    def run_params(self):
        with open(self.path + 'note.txt') as file:
            print(file.read())


class class2d_run:
    def __init__(self, data_folder, job_n):
        self.folder = data_folder
        self.path = self.folder + '/Class2D/{}/'.format(job_n)
        self.angles()
        self.cls_stats()
        # self.run_params()

    def angles(self):
        self.rot, self.tilt, self.psi = get_angles(self.path)

    def cls_stats(self):
        model_star = glob.glob(self.path + '/*model.star')
        model_star.sort(key=os.path.getmtime)
        self.model_star = model_star
        self.class_path, self.n_classes, self.iter_, self.class_dist_per_run, self.class_res_per_run = get_classes(
            self.path, self.model_star)

    def plot_stats_dist(self):
        self.cls_stats()
        plt.figure(figsize=(8, 4), dpi=100)

        for N, cls_ in enumerate(self.class_dist_per_run):
            plt.plot(range(0, len(cls_)), cls_.astype(float), label='{}'.format(N))

        plt.xlabel('Iter')
        plt.ylabel('Class distribution')
        plt.legend(ncol=10, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, shadow=True)

        plt.show()

    def plot_stats_res(self):
        self.cls_stats()
        plt.figure(figsize=(8, 4), dpi=100)

        for N, cls_ in enumerate(self.class_res_per_run):
            plt.plot(range(0, len(cls_)), cls_.astype(float), label='{}'.format(N))

        plt.xlabel('Iter')
        plt.ylabel('Resolution, A')
        plt.legend(ncol=10, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, shadow=True)
        plt.show()

    def plot_all_classes(self):
        self.cls_stats()

        display_2dclasses(images=self.get_2dcls(), model_star=self.model_star[-1], sort=True, columns=10, width=15)

    def get_2dcls(self):
        self.cls_stats()
        self.cls_list = get_class_list(self.class_path[-1])
        return self.cls_list

    def run_params(self):
        with open(self.path + 'note.txt') as file:
            print(file.read())


def resize_3d(volume_, new_size=100):
    # Otherwise is read only
    volume_ = volume_.copy()

    if new_size % 2 != 0:
        print('Box size has to be even!')
        quit()

    original_size = volume_.shape

    # Skip if volume is less than 100
    if original_size[0] <= 100:
        return volume_.copy()

    fft = np.fft.fftn(volume_)
    fft_shift = np.fft.fftshift(fft)

    # crop this part of the fft
    x1, x2 = int((volume_.shape[0] - new_size) / 2), volume_.shape[0] - int((volume_.shape[0] - new_size) / 2)

    fft_shift_new = fft_shift[x1:x2, x1:x2, x1:x2]

    # Apply spherical mask
    lx, ly, lz = fft_shift_new.shape
    X, Y, Z = np.ogrid[0:lx, 0:ly, 0:lz]
    dist_from_center = np.sqrt((X - lx / 2) ** 2 + (Y - ly / 2) ** 2 + (Z - lz / 2) ** 2)
    mask = dist_from_center <= lx / 2
    fft_shift_new[~mask] = 0

    fft_new = np.fft.ifftshift(fft_shift_new)
    new = np.fft.ifftn(fft_new)

    # Return only real part
    return new.real


def plot_volume(volume_, threshold=None, resize=None):
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    from skimage import measure
    # Resize volume if too big

    if type(volume_) == type(''):
        volume_ = mrcfile.open(volume_).data.copy()

    if resize:
        volume_ = resize_3d(volume_, new_size=resize)
    else:
        volume_ = volume_.copy()

    fig_ = go.Figure()

    # Here one could adjust the volume threshold if want to by adding level=level_value to marching_cubes
    try:
        if threshold != None:
            verts, faces, normals, values = measure.marching_cubes(volume_, threshold)
        else:
            verts, faces, normals, values = measure.marching_cubes(volume_)

    except RuntimeError:
        return None

    # Set the color of the surface based on the faces order. Here you can provide your own colouring
    color = np.zeros(len(faces))
    color[0] = 1  # because there has to be a colour range, 1st element is 1

    # create a plotly trisurf figure
    fig_volume = ff.create_trisurf(x=verts[:, 2],
                                   y=verts[:, 1],
                                   z=verts[:, 0],
                                   plot_edges=True,
                                   colormap=['rgb(120,150,180)'],
                                   simplices=faces,
                                   showbackground=False,
                                   show_colorbar=False,
                                   )

    fig_.add_trace(fig_volume['data'][0])
    fig_['layout'].update(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
        scene_dragmode='orbit')
    fig_.show()


def project_max_last_volume(path, pattern, dpi=150):
    last_volume = sorted(glob.glob(path + '{}.mrc'.format(pattern)))[-1]
    print(last_volume)

    volume = mrcfile.open(last_volume).data
    proj_list = []
    volume_shape = volume.shape[0]

    for axis in [0, 1, 2]:
        proj_list.append(np.max(volume, axis=axis))

    proj = np.concatenate((proj_list), axis=1)

    plt.figure(dpi=dpi)
    plt.imshow(proj, cmap='gray')
    plt.yticks([0.5 * volume_shape], ['projection'], rotation=90)
    plt.xticks([0.5 * volume_shape, 0.5 * volume_shape + volume_shape, 0.5 * volume_shape + volume_shape * 2],
               ['z', 'x', 'y'])

    plt.show()


def plot_rln_ref3d(folder, dpi=200):
    from plotly.subplots import make_subplots

    model_files = glob.glob(folder + "*model.star")
    model_files.sort(key=os.path.getmtime)

    n_inter = len(model_files)

    (class_paths, n_classes_, iter_, class_dist_, class_res_) = get_classes(folder, model_files)

    cls3d_projections = plot_3dclasses(class_paths, max_projection=True, joining_axes=[1, 0])

    shp_x = cls3d_projections.shape[1] / 3
    shp_y = cls3d_projections.shape[0]

    plt.figure(dpi=dpi)
    plt.imshow(cls3d_projections, cmap='gray')
    plt.yticks([0.5 * shp_y], ['projection'], rotation=90)
    plt.xticks([0.5 * shp_x, 0.5 * shp_x + shp_x, 0.5 * shp_x + shp_x * 2], ['Z', 'X', 'Y'])

    plt.show()

    import plotly.graph_objects as go
    fig_ = go.Figure()
    fig_['layout'].update(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))

    fig_ = go.Figure()

    for n, class_ in enumerate(class_res_):
        class_ = np.float16(class_)
        x = np.arange(0, class_res_.shape[1])

        fig_.add_scatter(x=x, y=class_, name='class {}'.format(n + 1), showlegend=False)

    fig_.update_xaxes(title_text="Iteration")
    fig_.update_yaxes(title_text="Class Resolution [A]")

    fig_.update_layout(
        title="Class Resolution [A]"
    )

    fig_.update_layout(
        autosize=False,
        height=400, )

    fig_.show()

    '''Angular dist plot'''
    fig_ = make_subplots(rows=1, cols=2)

    rot, tilt, psi = get_angles(folder)

    fig_.add_histogram2d(x=psi, y=tilt, showlegend=False, row=1, col=1,
                         xbins=dict(size=10), ybins=dict(size=10))

    fig_.update_xaxes(title_text="Psi [deg]", row=1, col=1)
    fig_.update_yaxes(title_text="Rotation [deg]", row=1, col=1)

    fig_.add_histogram2d(x=psi, y=rot, showlegend=False, row=1, col=2,
                         xbins=dict(size=10), ybins=dict(size=10))
    fig_.update_xaxes(title_text="Psi [deg]", row=1, col=2)
    fig_.update_yaxes(title_text="Tilt [deg]", row=1, col=2)

    fig_.update_layout(
        title="Angular distribution plots"
    )
    fig_.update_traces(showscale=False)

    fig_.update_layout(
        autosize=False,
        height=300, )

    fig_.show()


def plot_3dclasses(files, max_projection=False, joining_axes=[0, 1]):
    class_averages = []

    for n, class_ in enumerate(files):

        with mrcfile.open(class_) as mrc_stack:
            mrcs_file = mrc_stack.data
            z, x, y = mrc_stack.data.shape

            if max_projection:
                average_top = np.max(mrcs_file, axis=0)
                average_front = np.max(mrcs_file, axis=1)
                average_side = np.max(mrcs_file, axis=2)
            else:
                average_top = np.mean(mrcs_file, axis=0)
                average_front = np.mean(mrcs_file, axis=1)
                average_side = np.mean(mrcs_file, axis=2)

            # only a slice of the volume is plotted
            # for i in range(int(0.45 * z), int(0.55 * z)):

            average_class = np.concatenate((average_top, average_front, average_side), axis=joining_axes[0])
            class_averages.append(average_class)

    try:
        final_average = np.concatenate(class_averages, axis=joining_axes[1])

    except ValueError:
        final_average = []

    return final_average


def plot_rln_cls3d(folder):
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    model_files = glob.glob(folder + "*model.star")
    model_files.sort(key=os.path.getmtime)

    n_inter = len(model_files)

    (class_paths, n_classes_, iter_, class_dist_, class_res_) = get_classes(folder, model_files)

    cls3d_projections = plot_3dclasses(class_paths, max_projection=True)

    fig = px.imshow(cls3d_projections, binary_string=True,
                    labels=dict(x="Class", y="Projection axis"))
    fig.update_xaxes(side="top")

    labels_positions_x = np.linspace(1 / len(class_paths) * cls3d_projections.shape[1], cls3d_projections.shape[1],
                                     len(class_paths)) - 0.5 * 1 / len(class_paths) * cls3d_projections.shape[1]
    labels_x = ["Class {}<br>{}%".format(x, round(float(class_dist_[:, -1][x]) * 100, 2))
                for x, cls in enumerate(class_paths)]

    labels_positions_y = np.linspace(1 / 3 * cls3d_projections.shape[0], cls3d_projections.shape[0],
                                     3) - 0.5 * 1 / 3 * cls3d_projections.shape[0]
    labels_y = ["<b>Z </b>", "<b>X </b>", "<b>Y </b>"]

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=labels_positions_x,
            ticktext=labels_x
        )
    )

    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=labels_positions_y,
            ticktext=labels_y
        )
    )
    fig.write_image("fig1.pdf")
    fig.show()

    fig_ = go.Figure()

    for n, class_ in enumerate(class_dist_):
        class_ = np.float16(class_)
        x = np.arange(0, class_dist_.shape[1])

        fig_.add_scatter(x=x, y=class_, name='class {}'.format(n + 1), showlegend=False)

    fig_.update_xaxes(title_text="Iteration")
    fig_.update_yaxes(title_text="Class distribution")

    fig_.update_layout(
        title="Class distribution"
    )

    fig_.show()

    '''Class resolution plot'''

    fig_ = go.Figure()

    for n, class_ in enumerate(class_res_):
        class_ = np.float16(class_)
        x = np.arange(0, class_res_.shape[1])

        fig_.add_scatter(x=x, y=class_, name='class {}'.format(n + 1), showlegend=False)

    fig_.update_xaxes(title_text="Iteration")
    fig_.update_yaxes(title_text="Class Resolution [A]")

    fig_.update_layout(
        title="Class Resolution [A]"
    )

    fig_.show()

    '''Angular dist plot'''
    fig_ = make_subplots(rows=1, cols=2)

    rot, tilt, psi = get_angles(folder)

    fig_.add_histogram2d(x=psi, y=tilt, showlegend=False, row=1, col=1,
                         xbins=dict(size=10), ybins=dict(size=10))

    fig_.update_xaxes(title_text="Psi [deg]", row=1, col=1)
    fig_.update_yaxes(title_text="Rotation [deg]", row=1, col=1)

    fig_.add_histogram2d(x=psi, y=rot, showlegend=False, row=1, col=2,
                         xbins=dict(size=10), ybins=dict(size=10))
    fig_.update_xaxes(title_text="Psi [deg]", row=1, col=2)
    fig_.update_yaxes(title_text="Tilt [deg]", row=1, col=2)

    fig_.update_layout(
        title="Angular distribution plots"
    )
    fig_.update_traces(showscale=False)

    fig_.show()


def plot_postprocess(data_folder, jobname, dpi=150, plot_relevant=False,
                     res_ticks=(2.3, 3.0, 3.3, 4, 5, 7, 10, 20, 50)):
    '''Plot FSC curves from postprocess.star'''

    postprocess_star_path = '{}/PostProcess/{}/postprocess.star'.format(data_folder, jobname)
    postprocess_star_data = parse_star_whole(postprocess_star_path)
    res_ticks = np.array(res_ticks)

    fsc_data = postprocess_star_data['fsc']
    guinier_data = postprocess_star_data['guinier']

    fsc_x = fsc_data['_rlnAngstromResolution'].astype(float)
    if not plot_relevant:
        fsc_to_plot = ['_rlnFourierShellCorrelationCorrected', '_rlnFourierShellCorrelationUnmaskedMaps',
                       '_rlnFourierShellCorrelationMaskedMaps',
                       '_rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps']
    else:
        fsc_to_plot = ['_rlnFourierShellCorrelationUnmaskedMaps',
                       '_rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps']

    plt.figure(figsize=(8, 5), dpi=dpi)

    for meta in fsc_to_plot:
        # Limit the range of the data. Nobody needs 999A res. Start from ~50A?
        plt.plot(1 / fsc_x[8:], fsc_data[meta][8:].astype(float), label=meta.replace('_rln', ''))

    array = 1 / res_ticks

    plt.xticks(array, np.around(1 / array, 2))
    plt.axhline(0.5, linestyle='--', c='black')
    plt.axhline(0.143, linestyle='--', c='black')

    plt.annotate('0.143', (0.42, 0.16))
    plt.annotate('0.5', (0.42, 0.52))

    plt.legend(loc=1, fontsize=6)
    # plt.legend(bbox_to_anchor=(0.1, 1.3), loc='upper left')
    plt.xlabel('Resolution, Å', fontsize=15)
    plt.ylabel('FSC', fontsize=15)
    plt.xlim(1 / res_ticks[-1], 1 / np.array(fsc_x)[-1])

    plt.show()



def save_star_new(dicts_of_df, filename='out.star'):
    out_doc = cif.Document()

    for element in dicts_of_df.keys():
        out_particles = out_doc.add_new_block(element, pos=-1)

        # Row number is required for the column names to save the STAR file e.g. _rlnNrOfSignificantSamples #33
        column_names = dicts_of_df[element].columns
        column_names_to_star = []
        for n, name in enumerate(column_names):
            column_names_to_star.append(name + ' #{}'.format(n + 1))

        loop = out_particles.init_loop('', column_names_to_star)
        data_rows = dicts_of_df[element].to_numpy().astype(str).tolist()

        for row in data_rows:
            loop.add_row(row)

        out_doc.write_file(filename)
        # print('File "{}" saved.'.format(filename))


def plot_selected_classes(data_folder, select_job, columns_=6, width_=8):
    # For Relion4
    final_path = data_folder + '/Select/' + select_job
    select_star = parse_star(final_path + '/class_averages.star')[1]['_rlnReferenceImage']
    particles_star = parse_star(final_path + '/particles.star')[1]
    print('Selected {} particles'.format(particles_star.shape[0]))

    cls_imgs = []
    for cls_ in select_star:
        cls_data = cls_.split('@')
        cls_imgs.append(mrcfile.open(data_folder + '/' + cls_data[1]).data[int(cls_data[0]) - 1])

    display_2dclasses(cls_imgs, sort=True, columns=columns_, width=width_)


def plot_motioncorr(star, limit=100):
    motioncorr_data = parse_star_whole(star)['micrographs']

    if limit:
        motioncorr_data['_rlnAccumMotionLate'][motioncorr_data['_rlnAccumMotionLate'].astype(float) > limit] = limit
        motioncorr_data['_rlnAccumMotionEarly'][motioncorr_data['_rlnAccumMotionEarly'].astype(float) > limit] = limit
        motioncorr_data['_rlnAccumMotionTotal'][motioncorr_data['_rlnAccumMotionTotal'].astype(float) > limit] = limit

    plt.figure(figsize=(15, 10))
    plt_size_x = 2
    plt_size_y = 3

    plt.subplot(plt_size_x, plt_size_y, 4).set_title('Motion Correction')
    plt.hist2d(motioncorr_data['_rlnAccumMotionLate'].astype(float),
               motioncorr_data['_rlnAccumMotionEarly'].astype(float), bins=50, norm=LogNorm(), cmap='Blues')
    plt.ylabel('_rlnAccumMotionEarly')
    plt.xlabel('_rlnAccumMotionLate')

    plt.subplot(plt_size_x, plt_size_y, 1).set_title('Motion Correction')
    plt.plot(motioncorr_data['_rlnAccumMotionTotal'].astype(float))
    plt.title('_rlnAccumMotionTotal')

    plt.subplot(plt_size_x, plt_size_y, 2).set_title('Motion Correction')
    plt.plot(motioncorr_data['_rlnAccumMotionEarly'].astype(float))
    plt.title('_rlnAccumMotionEarly')

    plt.subplot(plt_size_x, plt_size_y, 3).set_title('Motion Correction')
    plt.plot(motioncorr_data['_rlnAccumMotionLate'].astype(float))
    plt.title('_rlnAccumMotionLate')

    plt.subplot(plt_size_x, plt_size_y, 5).set_title('Motion Correction')
    plt.hist(motioncorr_data['_rlnAccumMotionEarly'].astype(float), bins=50)
    plt.title('_rlnAccumMotionEarly')

    plt.subplot(plt_size_x, plt_size_y, 6).set_title('_rlnAccumMotionTotal')
    plt.hist(motioncorr_data['_rlnAccumMotionTotal'].astype(float), bins=50)
    plt.title('_rlnAccumMotionTotal')

    plt.show()


def plot_import(rln_folder, job_name, if_plotly=False):
    star_data = parse_star_whole(rln_folder + 'Import/' + job_name + '/movies.star')['movies']

    file_names = star_data['_rlnMicrographMovieName']

    '''Import by time'''

    file_mod_times = []
    for file in file_names:
        file_mod_times.append(datetime.datetime.fromtimestamp(os.path.getmtime(rln_folder + file)))

    if not if_plotly:
        plt.scatter(np.arange(0, len(file_mod_times)), file_mod_times, label='Time stamp')
        plt.title("Imported micrographs timeline")
        plt.ylabel("Time stamp")
        plt.xlabel("Index")
        plt.show()

    else:
        fig_ = go.Figure()

        fig_.add_scatter(x=np.arange(0, len(file_mod_times)), y=file_mod_times, name='Time stamp')

        fig_.update_xaxes(title_text="Index")
        fig_.update_yaxes(title_text="Time stamp")

        fig_.update_layout(
            title="Imported micrographs timeline"
        )
        fig_.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        fig_.show()


def plot_ctf_refine(data_folder, jobname, refine_job=''):
    ctf_data = glob.glob('{}/CtfRefine/{}/*.mrc'.format(data_folder, jobname))
    n_images = len(ctf_data)

    if n_images == 0:
        if refine_job != '':

            refine_data = parse_star_whole(refine_job)['particles']
            refine_data_U = refine_data['_rlnDefocusU'].values.astype(float)
            refine_data_V = refine_data['_rlnDefocusV'].values.astype(float)

            ctf_refine_data = \
            parse_star_whole('{}/CtfRefine/{}/particles_ctf_refine.star'.format(data_folder, jobname))['particles']
            ctf_refine_data_U = ctf_refine_data['_rlnDefocusU'].values.astype(float)
            ctf_refine_data_V = ctf_refine_data['_rlnDefocusV'].values.astype(float)

            ctf_U = refine_data_U - ctf_refine_data_U
            ctf_V = refine_data_V - ctf_refine_data_V

            plt.figure(figsize=(9, 3), dpi=100)
            plt.subplot(131)
            plt.title('_rlnDefocusU Change')
            plt.hist(ctf_U, bins=100)
            plt.ylabel('Count')
            plt.xlabel('Defocus change, Å')

            plt.subplot(132)
            plt.title('_rlnDefocusV Change')
            plt.hist(ctf_V, bins=100)
            plt.ylabel('Count')
            plt.xlabel('Defocus change, Å')

            plt.subplot(133)
            plt.title('_rlnDefocusU/V Change')
            plt.hist2d(ctf_U, ctf_V, bins=100)
            plt.xlabel('_rlnDefocusU change, Å')
            plt.ylabel('_rlnDefocusV change, Å')

            plt.tight_layout()
            plt.show()
            return



        else:
            print('Per particle CTF estimation. No Refine3D *data.star provided. Nothing to plot')
            return

    param_names = [os.path.basename(file).replace('.mrc', '') for file in ctf_data]
    ctf_images = [mrcfile.open(img).data for img in ctf_data]

    display_ctf_stats(ctf_images, param_names)


def display_ctf_stats(
        images, file_names,
        columns=2, width=8, height=5,
        label_wrap_length=50, label_font_size=8, label=True):
    import textwrap

    if file_names:
        labels = []
        for n, name in enumerate(file_names):
            labels.append(name)

    max_images = len(images)

    height = max(height, int(len(images) / columns) * height)
    plt.figure(figsize=(width, height))
    for i, image in enumerate(images):

        plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        plt.imshow(adjust_contrast(image, 5, 95), cmap='magma')

        if label:
            title = labels[i]
            title = textwrap.wrap(title, label_wrap_length)
            title = "\n".join(title)
            plt.title(title, fontsize=label_font_size);
        plt.axis('off')
        plt.tight_layout()


def plot_locres(data_folder, jobname, mask_job=''):
    locres_data = mrcfile.open('{}/LocalRes/{}/relion_locres.mrc'.format(data_folder, jobname)).data
    locres_map = mrcfile.open('{}/LocalRes/{}/relion_locres_filtered.mrc'.format(data_folder, jobname)).data
    data_shape = locres_data.shape

    if mask_job != '':
        mask = mrcfile.open('{}/MaskCreate/{}/mask.mrc'.format(data_folder, mask_job)).data.copy()
        mask[mask > 0] = 1
        locres_data = locres_data.copy() * mask

    plt.figure(figsize=(12, 3), dpi=150)

    plt.subplot(131)
    plt.imshow(locres_data[int(data_shape[0] / 2), :, :])
    cbar = plt.colorbar()
    cbar.set_label('Local Resolution, Å')
    plt.title('Slice Z')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(locres_data[:, int(data_shape[0] / 2)])
    cbar = plt.colorbar()
    cbar.set_label('Local Resolution, Å')
    plt.title('Slice X')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(locres_data[:, :, int(data_shape[0] / 2)])
    cbar = plt.colorbar()
    cbar.set_label('Local Resolution, Å')
    plt.title('Slice Y')
    plt.axis('off')