import os, sys, pickle, pprint, subprocess, time, random
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter, interpolation
from scipy.signal.windows import hann, gaussian
import cv2

import array_detect as ar


def get_lattice_vectors(
        calibration_name=None,
        bg=None,
        xPix=512,
        yPix=512,
        zPix=201,
        extent=5, # 寻找傅里叶尖峰时一个点的覆盖范围，需调整
        num_spikes=60, # 寻找傅里叶尖峰时的峰值数量，需调整
        tolerance=3., # 傅里叶基向量所得晶格点与尖峰对应的容差
        num_harmonics=3, # 傅里叶基向量的最小阶数
        show_ratio=1, # 显示傅里叶空间的峰值的图像比例，为了更好地看清低频点 0.25
        low_pass_filter=0.5, # 低通滤波的截止频率（高频有错位峰值）
        outlier_phase=1.,
        calibration_window_size=10,
        scan_type='visitech',
        scan_dimensions=None,
        verbose=True,
        display=True,
        animate=False, # 动画显示傅里叶空间的峰值寻找过程
        show_interpolation=False,
        show_calibration_steps=False,
        show_lattice=False,
        record_parameters=True):
    """
    由校准图像计算出照明晶格参数（给定一个扫描场图像栈，找出照明晶格图案的基向量。）
    :param calibration: 校准图像
    :param bg:
    :param use_lake_lattice:
    :param use_all_lake_parameters:
    :param xPix:
    :param yPix:
    :param zPix:
    :param bg_zPix:
    :param preframes:
    :param extent:
    :param num_spikes:
    :param tolerance:
    :param num_harmonics:
    :param outlier_phase:
    :param calibration_window_size:
    :param scan_type:
    :param scan_dimensions:
    :param verbose:
    :param display:
    :param animate:
    :param show_interpolation:
    :param show_calibration_steps:
    :param show_lattice:
    :param record_parameters:
    :return:
    """
    _, calibration_all = cv2.imreadmulti(calibration_name, flags=cv2.IMREAD_GRAYSCALE)
    calibration_all = np.array(calibration_all)

    # ar.detect_dot_centers(calibration_all, weighted=True, verbose=False, show=True)
    # 希望可以先得到粗略的向量，再进行精调

    print(" Detecting calibration illumination lattice parameters...")

    # 粗略估计晶格向量
    fft_data_folder, fft_abs, fft_avg = get_fft_abs(calibration_name, calibration_all)  # DC term at center
    filtered_fft_abs = spike_filter(fft_abs, display=False)

    # 在傅里叶域中寻找候选尖峰
    coords = find_spikes(fft_abs, filtered_fft_abs, extent=extent, num_spikes=num_spikes,low_pass_filter=0.5, show_ratio=show_ratio,display=display,
                         animate=animate)
    print(len(coords))

    # 用这些候选尖峰来确定傅里叶空间晶格
    if verbose:
        print("Finding Fourier-space lattice vectors...")
    basis_vectors = get_basis_vectors(fft_abs, coords, tolerance=tolerance, num_harmonics=num_harmonics,
                                      verbose=verbose)
    if verbose:
        print("Fourier-space lattice vectors:")
        for v in basis_vectors:
            print(v, "(Magnitude", np.sqrt((v ** 2).sum()), ")")

    # 通过约束傅里叶空间向量的和为零来修正这些向量。
    error_vector = sum(basis_vectors)
    corrected_basis_vectors = [v - ((1. / 3.) * error_vector) for v in basis_vectors]
    if verbose:
        print("Fourier-space lattice vector triangle sum:", error_vector)
        print("Corrected Fourier-space lattice vectors:")
        for v in corrected_basis_vectors:
            print(v)

    # 从傅里叶空间晶格确定实空间晶格
    area = np.cross(corrected_basis_vectors[0], corrected_basis_vectors[1])  # 平行四边形面积
    rotate_90 = ((0., -1.), (1., 0.))  # 逆时针旋转90度的旋转矩阵
    direct_lattice_vectors = [np.dot(v, rotate_90) * fft_abs.shape / area for v in corrected_basis_vectors]
    if verbose:
        print("Real-space lattice vectors:")
        for v in direct_lattice_vectors:
            print(v, "(Magnitude", np.sqrt((v ** 2).sum()), ")")
        print("Lattice vector triangle sum:")
        print(sum(direct_lattice_vectors))
        print("Unit cell area: (%0.2f)^2 square pixels" % (
            np.sqrt(np.abs(np.cross(direct_lattice_vectors[0], direct_lattice_vectors[1])))))

    # 看一下实空间中的基向量长什么样
    if display:
        show_lattice_overlay(calibration_all, direct_lattice_vectors, verbose=verbose)

    # 使用实空间中晶格向量和图像数据来测量（第一张校准图像）偏移向量
    offset_vector = get_offset_vector(
        image=calibration_all[0, :, :],
        direct_lattice_vectors=direct_lattice_vectors,
        verbose=verbose, display=display,
        show_interpolation=show_interpolation)

    # shift_vector = get_shift_vector(
    #     corrected_basis_vectors, fft_data_folder, filtered_fft_abs,
    #     num_harmonics=num_harmonics, outlier_phase=outlier_phase,
    #     verbose=verbose, display=display,
    #     scan_type=scan_type, scan_dimensions=scan_dimensions)
    #
    # corrected_shift_vector, final_offset_vector = get_precise_shift_vector(
    #     direct_lattice_vectors, shift_vector, offset_vector,
    #     image_data[-1, :, :], zPix, scan_type, verbose)
    #
    # if show_lattice:
    #     which_filename = 0
    #     while True:
    #         print("Displaying:", filename_list[which_filename])
    #         image_data = load_image_data(filename_list[which_filename])
    #         show_lattice_overlay(
    #             image_data, direct_lattice_vectors,
    #             offset_vector, corrected_shift_vector)
    #         if len(filename_list) > 1:
    #             which_filename = input(
    #                 "Display lattice overlay for which dataset? [done]:")
    #             try:
    #                 which_filename = int(which_filename)
    #             except ValueError:
    #                 if which_filename == '':
    #                     print("Done displaying lattice overlay.")
    #                     break
    #                 else:
    #                     continue
    #             if which_filename >= len(filename_list):
    #                 which_filename = len(filename_list) - 1
    #         else:
    #             break
    #
    # # image_data is large. Figures hold references to it, stinking up the place.
    # if display or show_lattice:
    #     plt.close('all')
    #     import gc
    #     gc.collect()  # Actually required, for once!
    #
    # if record_parameters:
    #     params_file_path = os.path.join(os.path.dirname(filename_list[0]), 'parameters.txt')
    #
    #     with open(params_file_path, 'w') as params:
    #         params.write("Direct lattice vectors: {}\n\n".format(repr(direct_lattice_vectors)))
    #         params.write("Corrected shift vector: {}\n\n".format(repr(corrected_shift_vector)))
    #         params.write("Offset vector: {}\n\n".format(repr(offset_vector)))
    #         try:
    #             params.write("Final offset vector: {}\n\n".format(repr(final_offset_vector)))
    #         except UnboundLocalError:
    #             params.write("Final offset vector: Not recorded\n\n")
    #         if lake is not None:
    #             params.write("Lake filename: {}\n\n".format(lake))
    #
    # if lake is None or bg is None:
    #     return direct_lattice_vectors, corrected_shift_vector, offset_vector
    # else:
    #     intensities_vs_galvo_position, background_frame = spot_intensity_vs_galvo_position(lake, xPix, yPix,
    #                                                                                        lake_lattice_vectors,
    #                                                                                        lake_shift_vector,
    #                                                                                        lake_offset_vector,
    #                                                                                        bg,
    #                                                                                        window_size=calibration_window_size,
    #                                                                                        show_steps=show_calibration_steps)
    #     return direct_lattice_vectors, corrected_shift_vector, offset_vector, intensities_vs_galvo_position, background_frame
    return


def show_lattice_overlay(calibration_all, direct_lattice_vectors, verbose=False):
    """
    展示 calibration_all 的第一张图片，并在图片上叠加原点（图像中点）、三个二维向量（从原点出发），叠加的图形用红色展示。

    :param calibration_all: 三维图堆栈
    :param direct_lattice_vectors: 三个二维向量
    :param verbose: 是否打印详细信息，默认为 False
    """
    # 获取第一张图片
    first_image = calibration_all[0]

    # 计算图像的中心点
    center_y, center_x = np.array(first_image.shape) // 2

    # 绘制第一张图片
    plt.imshow(first_image, cmap='gray')

    # 绘制原点
    plt.scatter(center_x, center_y, color='red', s=50)

    # 绘制三个二维向量
    for vector in direct_lattice_vectors:
        # 按坐标系论的xy，因此array的坐标顺序需要反一下
        plt.quiver(center_x, center_y, vector[1], vector[0], angles='xy', scale_units='xy', scale=1, color='red')

    # 设置坐标轴比例
    plt.axis('equal')

    # 显示图形
    plt.axis('off')
    plt.show()

    if verbose:
        print("Lattice overlay displayed successfully.")


# def detect_dot_centers(image, weighted=False, verbose=False, show=False):
#     """
#     检测点阵图像中所有点的质心。
#
#     :param image: 输入的灰度图像
#     :param weighted: 是否使用加权质心计算，默认为 False
#     :param verbose: 是否打印详细信息，默认为 False
#     :param show: 是否显示原始图像、二值图像和标记质心的图像，默认为 False
#     :return: 检测到的圆点质心列表
#     """
#     # 图像预处理：阈值处理
#     _, binary_image = cv2.threshold(image, 2 * image.min() / 3. + image.max() / 3., image.max(), cv2.THRESH_BINARY)
#     binary_image = binary_image.astype(np.uint8)  # 转换为uint8
#
#     dot_centers_stack = []
#     for binary_img in binary_image:
#         # 查找轮廓
#         contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         dot_centers = []
#         if ~weighted:
#             for contour in contours:
#                 # 计算轮廓的矩
#                 M = cv2.moments(contour)
#
#                 # 计算质心
#                 if M["m00"] != 0:
#                     cx = round(M["m10"] / M["m00"])
#                     cy = round(M["m01"] / M["m00"])
#                     dot_centers.append((cx, cy))
#         else:
#             for contour in contours:
#                 M = cv2.moments(contour)
#                 center = None
#                 if M["m00"] != 0:
#                     center = weighted_centroid(image, contour)
#                 if center is not None:
#                     dot_centers.append(center)
#         print("Detected %d dots centers" % len(dot_centers))
#         dot_centers_stack.append(dot_centers)
#
#     if show:
#         for index, dot_centers in enumerate(dot_centers_stack):
#             plt.figure(figsize=(16, 8))
#             plt.subplot(131)
#             plt.imshow(image[index], cmap='gray')
#             plt.title("Original Image")
#
#             plt.subplot(132)
#             plt.imshow(binary_image[index], cmap='gray')
#             plt.title("Binary Image")
#
#             color_image = cv2.cvtColor(image[index], cv2.COLOR_GRAY2BGR)
#             color_image = array_scale(color_image)
#             for center in dot_centers:
#                 cv2.circle(color_image, center, 0, (0, 0, 255), -1)  # 在质心位置画一个红色圆点
#
#             plt.subplot(133)
#             plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
#             if ~weighted:
#                 plt.title("Detected Unweighted Centers")
#             else:
#                 plt.title("Detected Weighted Centers")
#
#             plt.show()
#             flag = input("Continue?[y]/n: ")
#             # break
#
#     if verbose:
#         for i, center in enumerate(dot_centers):
#             print(f"Dot {i + 1} weighted center: ({center[0]}, {center[1]})")
#
#     return dot_centers


def get_fft_abs(filename, image_data, show_steps=False):
    """
    计算图像数据的傅里叶变换，并返回FFT的绝对值和平均值。
    如果之前已经对相同文件进行过FFT计算且结果文件存在，函数将直接加载这些结果，避免重复计算。

    :param filename: 输入图像文件名
    :param image_data: 输入图像数据
    :param show_steps: 是否显示每一步的结果，默认为 False
    :return: fft_data_folder, fft_abs, fft_avg
    """

    # 快速傅里叶变换（FFT）数据以一系列原始二进制文件的形式存储，每个二维z切片对应一个文件。这些文件的命名为000000.dat、000001.dat...
    basename = os.path.splitext(filename)[0]  # 去掉文件扩展名
    fft_abs_name = basename + '_fft_abs.npy'
    fft_avg_name = basename + '_fft_avg.npy'
    fft_data_folder = basename + '_fft_data'

    # 检查之前是否已经计算过相同文件的FFT结果，若存在则直接加载
    if (os.path.exists(fft_abs_name) and
            os.path.exists(fft_avg_name) and
            os.path.exists(fft_data_folder)):
        print("Loading", os.path.split(fft_abs_name)[1])  # 分离路径和文件名
        fft_abs = np.load(fft_abs_name)
        print("Loading", os.path.split(fft_avg_name)[1])
        fft_avg = np.load(fft_avg_name)
    else:
        # 不存在就生成
        print("Generating fft_abs, fft_avg and fft_data...")
        os.mkdir(fft_data_folder)
        fft_abs = np.zeros(image_data.shape[1:])
        fft_avg = np.zeros(image_data.shape[1:], dtype=np.complex128)
        window = (hann(image_data.shape[1]).reshape(image_data.shape[1], 1) *
                  hann(image_data.shape[2]).reshape(1, image_data.shape[2]))  # 汉宁窗
        if show_steps:
            plt.figure()
        for z in range(image_data.shape[0]):
            fft_data = np.fft.fftshift(  # Stored shifted!
                np.fft.fftn(window * image_data[z, :, :], axes=(0, 1)))
            fft_data.tofile(os.path.join(fft_data_folder, '%06i.dat' % (z)))
            fft_abs += np.abs(fft_data)
            if show_steps:
                plt.clf()
                plt.subplot(1, 3, 1)
                plt.title('Windowed slice %i' % z)
                plt.imshow(window * np.array(image_data[z, :, :]), cmap="gray", interpolation='nearest')
                plt.subplot(1, 3, 2)
                plt.title('FFT of slice %i' % z)
                plt.imshow(np.log(1 + np.abs(fft_data)), cmap="gray", interpolation='nearest')
                plt.subplot(1, 3, 3)
                plt.title("Cumulative sum of FFT absolute values")
                plt.imshow(np.log(1 + fft_abs), cmap="gray", interpolation='nearest')
                plt.show()
                input("Hit enter to continue...")
            fft_avg += fft_data
            sys.stdout.write('\rFourier transforming slice %i' % (z + 1))
            sys.stdout.flush()
        fft_avg = np.abs(fft_avg)
        np.save(fft_abs_name, fft_abs)
        np.save(fft_avg_name, fft_avg)

    return fft_data_folder, fft_abs, fft_avg


def spike_filter(fft_abs, display=False):
    """
    对傅里叶变换的绝对值进行滤波，以减少噪声并突出主要的峰值。
    :param fft_abs:
    :param display:
    :return:
    """
    # 高斯滤波，平滑处理
    f = gaussian_filter(np.log(1 + fft_abs), sigma=0.5)
    if display:
        display_image(f, 'Smoothed')

    # 水平方向滤波
    f = f - gaussian_filter(f, sigma=(0, 4))
    if display:
        display_image(f, 'Filtered left-right')

    # 垂直方向滤波
    f = f - gaussian_filter(f, sigma=(4, 0))
    if display:
        display_image(f, 'Filtered up-down')

    # 再次平滑处理
    f = gaussian_filter(f, sigma=0.5)
    f = gaussian_filter(f, sigma=(1.5))
    if display:
        display_image(f, 'Resmoothed')

    # 截断
    f = f * (f > 0)
    if display:
        display_image(f, 'Negative truncated')

    # 标准化处理
    f -= f.mean()
    f *= 1.0 / f.std()
    return f


def display_image(f, title):
    """
    显示图像并等待用户输入
    :param f: 要显示的图像数据
    :param title: 图像的标题
    """
    plt.imshow(f, cmap="gray", interpolation='nearest')
    plt.title(title)
    plt.show()


def find_spikes(fft_abs, filtered_fft_abs, extent=15, num_spikes=300, low_pass_filter=0.5,show_ratio=1., display=True, animate=False):
    """
    查找傅里叶变换的绝对值中最大的峰值，这些峰值通常对应于图像中的亮点。
    :param fft_abs: 傅里叶变换的绝对值之和
    :param filtered_fft_abs: 滤波后的傅里叶变换的绝对值之和
    :param extent: 峰值的搜索范围
    :param num_spikes: 查找峰值的最大次数
    :param display: 是否显示fft_abs 和 filtered_fft_abs 的图像
    :param animate: 是否显示查找过程
    :return:
    """
    center_pix = np.array(fft_abs.shape) // 2
    log_fft_abs = np.log(1 + fft_abs)
    filtered_fft_abs = np.array(filtered_fft_abs)
    filtered_fft_abs[0:int(low_pass_filter*filtered_fft_abs.shape[0]/2),:] = 0
    filtered_fft_abs[int(-low_pass_filter*filtered_fft_abs.shape[0]/2):,:] = 0
    filtered_fft_abs[:, 0:int(low_pass_filter * filtered_fft_abs.shape[1]/2)] = 0
    filtered_fft_abs[:, int(-low_pass_filter * filtered_fft_abs.shape[1]/2):] = 0

    if display:
        # 截取fft_abs和 filtered_fft_abs的中心区域
        log_fft_abs_show = log_fft_abs[int(center_pix[0] - center_pix[0] * show_ratio): int(
            center_pix[0] + center_pix[0] * show_ratio),
                           int(center_pix[1] - center_pix[1] * show_ratio): int(
                               center_pix[1] + center_pix[1] * show_ratio)]
        filtered_fft_abs_show = filtered_fft_abs[int(center_pix[0] - center_pix[0] * show_ratio):int(
            center_pix[0] + center_pix[0] * show_ratio),
                                int(center_pix[1] - center_pix[1] * show_ratio): int(
                                    center_pix[1] + center_pix[1] * show_ratio)]

        # 显示 fft_abs 和 filtered_fft_abs 的图像
        image_extent = np.float64([-0.5 - show_ratio * center_pix[1],
                                   filtered_fft_abs.shape[1] - 0.5 - (2-show_ratio) * center_pix[1],
                                   filtered_fft_abs.shape[0] - 0.5 - (2-show_ratio) * center_pix[0],
                                   -0.5 - show_ratio * center_pix[0]])  # 左边界、右边界、下边界、上边界（以图像中心为原点）（只是数值，不会截取图像）
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(log_fft_abs_show, cmap="gray", interpolation='nearest', extent=image_extent)
        plt.title('Average Fourier magnitude')
        plt.subplot(1, 2, 2)
        plt.imshow(np.array(filtered_fft_abs_show), cmap="gray", interpolation='nearest', extent=image_extent)
        plt.title('Filtered average Fourier magnitude')
        plt.show()

    coords = []  # 储存尖峰的坐标
    if animate:
        plt.figure()
        print('Center pixel:', center_pix)
    for i in range(num_spikes):

        print(np.array(np.unravel_index(filtered_fft_abs.argmax(), filtered_fft_abs.shape)),filtered_fft_abs.max())
        cv2.imwrite("filtered_fft_abs.png", filtered_fft_abs*255/filtered_fft_abs.max())
        coords.append(np.array(np.unravel_index(filtered_fft_abs.argmax(), filtered_fft_abs.shape)))
        c = coords[-1]
        # 将当前尖峰周围的区域置为0，避免重复检测
        xSl = slice(max(c[0] - extent, 0), min(c[0] + extent, filtered_fft_abs.shape[0]))
        ySl = slice(max(c[1] - extent, 0), min(c[1] + extent, filtered_fft_abs.shape[1]))

        filtered_fft_abs[xSl, ySl] = 0

        if animate:
            # 截取filtered_fft_abs的中心区域
            filtered_fft_abs_show = filtered_fft_abs[int(center_pix[0] - center_pix[0] * show_ratio):int(
                center_pix[0] + center_pix[0] * show_ratio),
                                    int(center_pix[1] - center_pix[1] * show_ratio): int(
                                        center_pix[1] + center_pix[1] * show_ratio)]

            image_extent = np.float64([0,filtered_fft_abs.shape[1]*show_ratio,
                                       filtered_fft_abs.shape[0]*show_ratio,0])
                                        # 左边界、右边界、下边界、上边界（以图像中心为原点）（只是数值，不会截取图像）

            print(i, ':', c)
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.imshow(filtered_fft_abs_show, cmap="gray", interpolation='nearest', extent=image_extent)
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.plot(filtered_fft_abs_show.max(axis=1))
            plt.show()
            if i == 0:
                input('.')

    coords = [c - center_pix for c in coords]  # 将所有尖峰的坐标转换为相对于图像中心的坐标
    coords = sorted(coords, key=lambda x: x[0] ** 2 + x[1] ** 2)  # 按向量幅度从小到大排序尖峰的坐标

    return coords


def get_basis_vectors(fft_abs, coords, tolerance=3., num_harmonics=3, verbose=False):
    """
    从傅里叶变换的绝对值中找到一组基本向量，这些向量可能对应于图像中的晶格。
    :param fft_abs: 傅里叶变换的绝对值
    :param coords: 尖峰的坐标
    :param tolerance: 查找晶格点时允许的误差容限
    :param num_harmonics: 所需的谐波数量，用于判断是否找到足够的晶格点
    :param verbose: 是否打印详细的调试信息
    :return: 晶格基向量
    """
    for i in range(len(coords)):  # Where to start looking.
        basis_vectors = []
        for c, coord in enumerate(coords):
            if c < i:
                continue

            # 中心峰值
            if c == 0:
                if max(abs(coord)) > 0:  # 第一个最大值不在中心
                    print("c:", c)
                    print("Coord:", coord)
                    print("Coordinates:")
                    for x in coords:
                        print(x)
                    raise UserWarning('No peak at the central pixel')
                else:
                    continue

            if coord[0] < 0 or (coord[0] == 0 and coord[1] < 0):
                # Ignore the negative versions
                if verbose:
                    print("\nIgnoring:", coord)
            else:
                # Check for harmonics
                if verbose:
                    print("\nTesting:", coord)
                num_vectors, points_found = test_basis(coords, [coord], tolerance=tolerance, verbose=verbose)
                if num_vectors > num_harmonics:
                    # 找到了足够的谐波，目前先保留它
                    basis_vectors.append(coord)
                    # center_pix = np.array(fft_abs.shape) // 2
                    # furthest_spike = points_found[-1] + center_pix
                    if verbose:
                        print("Appending", coord)
                        print("%i harmonics found, at:" % (num_vectors - 1))
                        for p in points_found:
                            print(' ', p)

                    # 如果向量单独测试通过了，就需要通过组合测试
                    if len(basis_vectors) > 1:
                        if verbose:
                            print("\nTesting combinations:", basis_vectors)
                        num_vectors, points_found = test_basis(coords, basis_vectors, tolerance=tolerance,
                                                               verbose=verbose)
                        if num_vectors > num_harmonics:
                            # 找到了足够的谐波，组合通过测试
                            if len(basis_vectors) == 3:
                                # 找到三个基向量，则完成任务，查找更准确的基向量
                                precise_basis_vectors = get_precise_basis(coords, basis_vectors, fft_abs,
                                                                          tolerance=tolerance, verbose=verbose)
                                (x_1, x_2, x_3) = sorted(precise_basis_vectors, key=lambda x: abs(x[0]))  # 按元素绝对值大小排序
                                possibilities = sorted(
                                    ([x_1, x_2, x_3],
                                     [x_1, x_2, -x_3],
                                     [x_1, -x_2, x_3],
                                     [x_1, -x_2, -x_3]),
                                    key=lambda x: (np.array(sum(x)) ** 2).sum()
                                )  # 最终结果从小到大排序

                                if verbose:
                                    print("Possible triangle combinations:")
                                    for p in possibilities:
                                        print(" ", p)

                                precise_basis_vectors = possibilities[0]  # 取排序最小的那个
                                if precise_basis_vectors[-1][0] < 0:
                                    for p in range(3):
                                        precise_basis_vectors[p] *= -1
                                return precise_basis_vectors

                        # 组合测试未通过，删除最后进来的向量
                        else:
                            # Blame the new guy, for now.
                            basis_vectors.pop()
    else:
        raise UserWarning(
            "Basis vector search failed. Diagnose by running with verbose=True")


def test_basis(coords, basis_vectors, tolerance, verbose=False):
    """
    查找预期的晶格，返回找到的点，并在失败时停止搜索
    :param coords: 傅里叶图像中的峰值坐标相对于中心原点的向量
    :param basis_vectors: [coord]，或多个通过测试的向量的集合
    :param tolerance:
    :param verbose:
    :return:
    """
    points_found = list(basis_vectors)
    num_vectors = 2
    searching = True
    while searching:
        # 生成基向量的所有可能的组合（允许重复），并将这些组合的和作为预期的晶格点
        if verbose:
            print("Looking for combinations of %i basis vectors." % num_vectors)
        lattice = [sum(c) for c in combinations_with_replacement(basis_vectors, num_vectors)]
        if verbose:
            print("Expected lattice points:", lattice)

        for i, lat in enumerate(lattice):
            for c in coords:
                dif = np.sqrt(((lat - c) ** 2).sum())
                if dif < tolerance:
                    if verbose:
                        print("Found lattice point:", c)
                        print(" Distance:", dif)
                        if len(basis_vectors) == 1:
                            print(" Fundamental:", c * 1.0 / num_vectors)
                    points_found.append(c)
                    break
            else:  # 如果没有找到预期的晶格点，停止搜索
                if verbose:
                    print("Expected lattice point not found")
                searching = False
        if not searching:
            return num_vectors, points_found
        # 增加基向量的组合数量，继续下一轮搜索
        num_vectors += 1


def get_precise_basis(coords, basis_vectors, fft_abs, tolerance, verbose=False):
    """
    使用预期的晶格来估计基向量的精确值。

    :param coords: 尖峰的坐标列表
    :param basis_vectors: 初步的基向量列表
    :param fft_abs: 傅里叶变换的绝对值数组
    :param tolerance: 查找晶格点时允许的误差容限
    :param xPix: 图像的水平像素数，默认为 128
    :param yPix: 图像的垂直像素数，默认为 128
    :param verbose: 是否打印详细的调试信息，默认为 False
    :return: 精确的基向量数组
    """
    if verbose:
        print("\nAdjusting basis vectors to match lattice...")
    center_pix = np.array(fft_abs.shape) // 2
    basis_vectors = list(basis_vectors)
    spike_indices = []  # 晶格点索引
    spike_locations = []  # 晶格点位置

    num_vectors = 2
    searching = True
    while searching:
        # 下面步骤的结果依赖于两次调用函数给出的组合顺序相同
        combinations = [c for c in combinations_with_replacement(basis_vectors, num_vectors)]  # 基向量组和
        combination_indices = [c for c in combinations_with_replacement((0, 1, 2), num_vectors)]  # 索引组合

        for i, comb in enumerate(combinations):
            lat = sum(comb)  # 计算组合的和，得到预期的晶格点
            key = tuple([combination_indices[i].count(v) for v in (0, 1, 2)])  # 计算0、1、2出现的次数

            for c in coords:
                dif = np.sqrt(((lat - c) ** 2).sum())
                if dif < tolerance:  # 寻找对应晶格点
                    true_max = None
                    p = c + center_pix  # 将尖峰坐标转换为图像中的实际坐标
                    # 检查坐标是否在图像范围内
                    if 0 < p[0] < fft_abs.shape[0] and 0 < p[1] < fft_abs.shape[0]:
                        # 使用 simple_max_finder 函数估计精确的最大值位置（用在空域上是不是可以改为周围一定范围内的质心）
                        true_max = c + simple_max_finder(fft_abs[p[0] - 1:p[0] + 2, p[1] - 1:p[1] + 2],
                                                         show_plots=False)
                    if verbose:
                        print("Found lattice point:", c)
                        print("Estimated position:", true_max)
                        print("Lattice index:", key)

                    spike_indices.append(key)  # 记录晶格点的索引
                    spike_locations.append(true_max)  # 记录晶格点的精确位置
                    break
            else:  # 没有找到预期的晶格点
                if verbose:
                    print("Expected lattice point not found")
                searching = False
        if not searching:  # 根据找到的尖峰，估计基向量
            A = np.array(spike_indices)  # 晶格点索引矩阵
            v = np.array(spike_locations)  # 晶格点位置矩阵
            # 使用最小二乘法求解精确的基向量（Ax=v）
            print(A.shape,A.dtype, v.shape,v.dtype)
            precise_basis_vectors, residues, rank, s = np.linalg.lstsq(A, v, rcond=None)
            if verbose:
                print("Precise basis vectors:")
                print(precise_basis_vectors)
                print("Residues:", residues)
                print("Rank:", rank)
                print("s:", s)
                print()
            return precise_basis_vectors
        # 增加基向量的组合数量，继续下一轮搜索
        num_vectors += 1


def combinations_with_replacement(iterable, r):
    """
    用于生成可重复的组合。与普通的组合不同，可重复组合允许元素在组合中重复出现。例如，对于集合 ['a', 'b', 'c']，
    选取 2 个元素的可重复组合包括 ('a', 'a')、('a', 'b') 等。
    print([i for i in combinations_with_replacement(['a', 'b', 'c'], 2)])
    [('a', 'a'), ('a', 'b'), ('a', 'c'), ('b', 'b'), ('b', 'c'), ('c', 'c')]
    :param iterable: 元素列表
    :param r: 组合个数
    :return:
    """
    """

    """
    pool = tuple(iterable)
    n = len(pool)
    for indices in product(range(n), repeat=r):
        if sorted(indices) == list(indices):
            yield tuple(pool[i] for i in indices)


def simple_max_finder(a, show_plots=True):
    """Given a 3x3 array with the maximum pixel in the center,
    estimates the x/y position of the true maximum"""
    true_max = []
    inter_points = np.arange(-1, 2)
    for data in (a[:, 1], a[1, :]):
        my_fit = np.poly1d(np.polyfit(inter_points, data, deg=2))
        true_max.append(-my_fit[1] / (2.0 * my_fit[2]))

    true_max = np.array(true_max)

    if show_plots:
        print("Correction:", true_max)
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(a, interpolation='nearest', cmap="gray")
        plt.axhline(y=1 + true_max[0])
        plt.axvline(x=1 + true_max[1])
        plt.subplot(1, 3, 2)
        plt.plot(a[:, 1])
        plt.axvline(x=1 + true_max[0])
        plt.subplot(1, 3, 3)
        plt.plot(a[1, :])
        plt.axvline(x=1 + true_max[1])
        plt.show()

    return true_max

def get_offset_vector(image, direct_lattice_vectors, prefilter='median', filter_size = 3,verbose=True, display=True,
                      show_interpolation=True):
    """
    已知晶格向量，计算一张图片中晶格点的偏移向量？
    :param image:
    :param direct_lattice_vectors:
    :param prefilter:
    :param filter_size:
    :param verbose:
    :param display:
    :param show_interpolation:
    :return:
    """
    # 中值滤波
    if prefilter == 'median':
        image = median_filter(image, size=filter_size)

    if verbose:
        print("\nCalculating offset vector...")

    # 窗口大小：晶格向量x\y方向上的最大距离，加上一个缓冲区（2）
    ws = 2 + int(max([abs(v).max() for v in direct_lattice_vectors]))
    if verbose:
        print("Window size:", ws)

    # 按照窗口大小初始化窗口，shape=(2 * ws + 1, 2 * ws + 1)
    window = np.zeros([2 * ws + 1] * 2, dtype=np.float64)
    lattice_points = generate_lattice(image.shape, direct_lattice_vectors, edge_buffer=2 + ws)
    for lp in lattice_points:
        window += get_centered_subimage(center_point=lp, window_size=ws, image=image.astype(float))

    if display:
        plt.figure()
        plt.imshow(window, interpolation='nearest', cmap="gray")
        plt.title('Lattice average\nThis should look like round blobs')
        plt.show()

    buffered_window = np.array(window)
    buffered_window[:2, :] = 0
    buffered_window[-2:, :] = 0
    buffered_window[:, :2] = 0
    buffered_window[:, -2:] = 0

    while True:  # Don't want maxima on the edges
        max_pix = np.unravel_index(buffered_window.argmax(), window.shape)
        if (3 < max_pix[0] < window.shape[0] - 3) and (3 < max_pix[1] < window.shape[1] - 3):
            break
        else:
            buffered_window = gaussian_filter(buffered_window, sigma=2)

    if verbose:
        print("Maximum pixel in lattice average:", max_pix)

    correction = simple_max_finder(
        window[max_pix[0] - 1:max_pix[0] + 2, max_pix[1] - 1:max_pix[1] + 2],
        show_plots=show_interpolation)

    offset_vector = max_pix + correction + np.array(image.shape) // 2 - ws
    if verbose:
        print("Offset vector:", offset_vector)

    return offset_vector

def generate_lattice(image_shape, lattice_vectors, center_pix='image', edge_buffer=2, return_i_j=False):
    # 修正判断
    if isinstance(center_pix, str):
        if center_pix == 'image':
            center_pix = np.array(image_shape) // 2
    else:
        center_pix = np.array(center_pix) - (np.array(image_shape) // 2)
        lattice_components = np.linalg.solve(np.vstack(lattice_vectors[:2]).T, center_pix)
        lattice_components_centered = np.mod(lattice_components, 1)
        lattice_shift = lattice_components - lattice_components_centered
        center_pix = (lattice_vectors[0] * lattice_components_centered[0] +
                      lattice_vectors[1] * lattice_components_centered[1] +
                      np.array(image_shape) // 2)

    num_vectors = int(np.round(1.5 * max(image_shape) / np.sqrt((lattice_vectors[0] ** 2).sum())))  # changed
    lower_bounds = (edge_buffer, edge_buffer)
    upper_bounds = (image_shape[0] - edge_buffer, image_shape[1] - edge_buffer)
    i, j = np.mgrid[-num_vectors:num_vectors, -num_vectors:num_vectors]
    i = i.reshape(i.size, 1)
    j = j.reshape(j.size, 1)
    lp = i * lattice_vectors[0] + j * lattice_vectors[1] + center_pix
    valid = np.all(lower_bounds < lp, 1) * np.all(lp < upper_bounds, 1)
    lattice_points = list(lp[valid])
    if return_i_j:
        return (lattice_points,
                list(i[valid] - lattice_shift[0]),
                list(j[valid] - lattice_shift[1]))
    else:
        return lattice_points

def get_centered_subimage(
        center_point, window_size, image, background='none'):
    x, y = np.round(center_point).astype(int)
    xSl = slice(max(x - window_size - 1, 0), x + window_size + 2)
    ySl = slice(max(y - window_size - 1, 0), y + window_size + 2)
    subimage = np.array(image[xSl, ySl])

    if not isinstance(background, str):
        subimage -= background[xSl, ySl]
    interpolation.shift(subimage, shift=(x, y) - center_point, output=subimage)
    return subimage[1:-1, 1:-1]

