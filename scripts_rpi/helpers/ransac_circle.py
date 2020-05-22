import numpy as np
import math
import matplotlib.pyplot as plt

"""
Great Tutorial: https://www.uio.no/studier/emner/matnat/its/nedlagte-emner/UNIK4690/v16/forelesninger/lecture_3_3-robust-estimation-with-ransac.pdf
Source of calc_circle_params: https://github.com/Yiphy/Ransac-2d-Shape-Detection/blob/master/ransac_circle2d.cpp
- Using least-squares doesn't work because heavily affected by outliers since all points are weighed equally.
- General Idea of Ransac:
1. Estimate a circle from n-points, where n is small as possible
2. Determine which points from set are inliers(good match) for the estimated circles
- Circle only needs n = 3 points(x,y), determined by calc_circle_params(pts)
"""

def run_ransac(points, max_iters=50, inlier_thresh=5):
    N = 3  # need three points to form circle
    M = points.shape[0]  # number of points
    best_inliers = None
    best_backup = None
    most_inliers = 0
    
    for i in range(max_iters):
        sample = np.random.randint(low=0, high=M, size=N)
        try:
            center, r = calc_circle_params(points[sample])
        except ZeroDivisionError:
            continue
        # print("~Center: (%.2f, %.2f), ~R: %.2f" % (
        #     center[0], center[1], r))
        inliers = find_inliers(points, center, r, inlier_thresh)
        if inliers.shape[0] > most_inliers:
            most_inliers = inliers.shape[0]
            best_inliers = inliers
            best_backup = (center, r)

    confidence = most_inliers / M
    try:
        return least_squares_fit(best_inliers), confidence
    except:
        return best_backup, confidence
    

def least_squares_fit(inliers):
    # eigenvalue decomposition to find least-squares match of params
    # Ax = b, solve for x = [p1, p2, p3]
    # Source: link at top, slide 15
    M = inliers.shape[0]
    A = np.concatenate([inliers, np.ones((M,1))], axis=1)  # [x, y, 1]
    b = inliers[:,0]**2 + inliers[:,1]**2
    x = np.linalg.inv(A.T @ A) @ A.T @ b
    cx = x[0] / 2
    cy = x[1] / 2
    center = np.array([cx,cy])
    r = math.sqrt(x[2] + cx**2 + cy**2)
    return center, r


def find_inliers(points, center, r, threshold=5):
    distances = np.linalg.norm(points - center, axis=1)
    is_inlier = abs(distances - r) < threshold
    inliers = points[is_inlier]
    return inliers


def calc_circle_params(pts: np.array):
    assert(pts.shape == (3, 2))  # 3 points (x,y)
    a, b, c = get_pair_params(pts[0,:], pts[1,:])
    d, e, f = get_pair_params(pts[1,:], pts[2,:])
    cx = (b*f - e*c) / (b*d - e*a)
    cy = (d*c - a*f) / (b*d - e*a)
    center = np.array([cx,cy])
    r = np.linalg.norm(pts[0,:] - center)
    return center, r


def get_pair_params(pt1, pt2):
    x1,y1 = pt1
    x2,y2 = pt2
    a = 2*(x2-x1)
    b = 2*(y2-y1)
    c = x2**2 + y2**2 - x1**2 - y1**2
    return (a,b,c)


def gen_true_inliers(center, r, N=30):
    rand_thetas = np.random.random_sample(size=(N,1)) * 2*math.pi
    pts = center + np.hstack([np.cos(rand_thetas), np.sin(rand_thetas)]) * r
    return pts

def test_ransac(visualize=True):
    # generate points along circle with some random points
    N = 30  # number of inliers
    M = 10  # number of noisy points
    rand_cx = (np.random.random() - 0.5) * 100
    rand_cy = (np.random.random() - 0.5) * 100
    true_center = np.array([rand_cx, rand_cy])
    true_r = np.random.random() * 50
    print("True Center: (%.2f, %.2f), True R: %.2f" % (true_center[0], true_center[1], true_r))
    true_inliers = gen_true_inliers(true_center, true_r, N)
    rand_noise = (np.random.random_sample(size=(M,2)) - 0.5) * 150
    all_pts = np.vstack([true_inliers, rand_noise])
    np.random.shuffle(all_pts)
    # np.savez('vars', all_pts=all_pts, true_center=true_center, true_r=true_r)
    # data = np.load("scripts_rpi/helpers/failure.npz")
    # all_pts=data["all_pts"]
    # true_center=data["true_center"]
    # true_r=data["true_r"]
    # print("True Center: (%.2f, %.2f), True R: %.2f" % (true_center[0], true_center[1], true_r))

    # run ransac
    threshold = true_r * 0.05  # 5% within true radius
    approx_center, approx_r, confidence = run_ransac(all_pts, inlier_thresh=threshold)
    print("~Center: (%.2f, %.2f), ~R: %.2f" % (
        approx_center[0], approx_center[1], approx_r))

    # visualize results
    if visualize:
        fig, ax = plt.subplots()
        min_x, min_y = np.min(all_pts, axis=0)
        max_x, max_y = np.max(all_pts, axis=0)
        ax.set_xlim((min_x, max_x))
        ax.set_ylim((min_y, max_y))
        ax.scatter(all_pts[:,0], all_pts[:,1])
        true_circle = plt.Circle(true_center, true_r, 
            color='r', fill=False, linewidth=2)
        approx_circle = plt.Circle(approx_center, approx_r, 
            color='b', fill=False, linewidth=2)
        ax.add_artist(true_circle)
        ax.add_artist(approx_circle)
        plt.show()

    return np.linalg.norm(approx_center - true_center), true_r - approx_r


if __name__== "__main__":
    avg_center_err = 0
    avg_rad_err = 0
    for i in range(20):
        center_err, rad_err = test_ransac(visualize=False)
        avg_center_err += center_err
        avg_rad_err += rad_err

    print("Avg center error: %.2f" % (avg_rad_err / 20))
    print("Avg radius error: %.2f" % (avg_rad_err / 20))