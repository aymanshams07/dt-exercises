
from collections import OrderedDict
from scipy.stats import multivariate_normal
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from math import floor, sqrt



class LaneFilterHistogramKF():
    """ Generates an estimate of the lane pose.

    TODO: Fill in the details

    Args:
        configuration (:obj:`List`): A list of the parameters for the filter

    """

    def __init__(self, **kwargs):
        param_names = [
            # TODO all the parameters in the default.yaml should be listed here.
            'mean_d_0',
            'mean_phi_0',
            'sigma_d_0',
            'sigma_phi_0',
            'delta_d',
            'delta_phi',
            'd_max',
            'd_min',
            'phi_max',
            'phi_min',
            'cov_v',
            'linewidth_white',
            'linewidth_yellow',
            'lanewidth',
            'min_max',
            'sigma_d_mask',
            'sigma_phi_mask',
            'range_min',
            'range_est',
            'range_max',
        ]

        for p_name in param_names:
            assert p_name in kwargs
            setattr(self, p_name, kwargs[p_name])



        self.encoder_resolution = 0
        self.wheel_radius = 0.0
        self.baseline = 0.0
        self.initialized = False
        self.reset()

    def reset(self):
        self.mean_0 = [self.mean_d_0, self.mean_phi_0]
        self.cov_0 = [[self.sigma_d_0, 0], [0, self.sigma_phi_0]]
        self.belief = {'mean': self.mean_0, 'covariance': self.cov_0}

    def predict(self, dt, left_encoder_delta, right_encoder_delta):
        #TODO update self.belief based on right and left encoder data + kinematics
        del_sr = (right_encoder_delta / 135) * (2 * 3.14 * 0.0318)
        mean_del_sr = np.mean(del_sr)
        del_sl = (left_encoder_delta / 135) * (2 * 3.14 * 0.0318)
        mean_del_sl = np.mean(del_sl)
        axel_len = 0.1
        del_theta = (del_sr - del_sl) / axel_len
        del_s = (del_sr + del_sl) / 2
        initial_y = self.sigma_d_0
        initial_theta = self.sigma_phi_0

        # calculating the components
        del_x = del_s * np.cos(initial_theta + (del_theta / 2))
        del_y = del_s * np.sin(initial_theta + (del_theta / 2))
        u_t = (np.array([[del_y, del_theta]])).T
        meanu_t = self.mean_0
        mu_t = (np.array([meanu_t])).T
        A = np.array([[del_y, 0], [0, del_theta]])
        print('this is shape of A', A.shape)
        A_gauss = gaussian_filter(A, sigma=0.5)
        print('this is shape of gaussian of A', A_gauss.shape)
        # A = np.array([[1/1-left_slip_ratio, 0], [0, 1/1-right_slip_ratio]])
        # B matrix with gaussian filter
        B = np.array([[0, -del_y], [del_theta, 0]])
        B_gauss = gaussian_filter(B, sigma=0.5)

        ## Predict mu step
        predicted_mu = A_gauss @ mu_t + B @ u_t

        # start from here to calculate jaco
        k = 27
        sigma_sr = (k * abs(del_sr))**2
        sigma_sl = (k * abs(del_sl))**2
        c_x = np.array([[sigma_sr, 0], [0, sigma_sl]])

        # jaco_1 = np.array([[0, -del_y], [1, del_x], [0, 1]])
        jaco_1 = np.array([[0, -del_y], [0, 1]])


        p_1 = 0.5 * np.cos(initial_theta + del_theta / 2)
        p_2 = 0.5 * np.sin(initial_theta + del_theta / 2)
        p_3 = (del_s / 2) * (1 / axel_len) * np.cos(initial_theta + del_theta / 2)
        p_4 = (del_s / 2) * (1 / axel_len) * np.sin(initial_theta + del_theta / 2)

        # jaco_2 = np.array([[(p_1 - p_4), (p_1 + p_4)], [(p_2 + p_3), (p_2 - p_3)], [(1 / axel_len), (-1 / axel_len)]])
        jaco_2 = np.array([[(p_2 + p_3), (p_2 - p_3)], [(1 / axel_len), (-1 / axel_len)]])

        # work on this (initial covariance matrix)
        c_x_1 = np.array([[p_2, 0], [p_4, 0]])
        print('this is shape of c_x_1', c_x_1.shape)

        # putting it all together
        c_y = jaco_1 @ c_x_1 @ jaco_1.T + jaco_2 @ c_x @ jaco_2.T

        # this is motion noise
        #Q = np.array([[0.3, 0], [0, 0.3]])
        Q = gaussian_filter(c_y, sigma=0.5)

        # predict covariance step
        predicted_sigma = A @ c_y @ A.T + Q
        print('this is the predicted sigma', predicted_sigma)
        print('this is the predicted sigma shape', Q.shape)

        self.belief = {'mean': predicted_mu, 'covariance': predicted_sigma}
        print('this is the current belief', self.belief)

        if not self.initialized:
            return

    def update(self, segments):
        # prepare the segments for each belief array
        segmentsArray = self.prepareSegments(segments)

        # generate all belief arrays
        measurement_likelihood = self.generate_measurement_likelihood(
            segmentsArray)
        #print('this is measurement likelihood', measurement_likelihood)
        # min_meas = np.min(measurement_likelihood[np.nonzero(measurement_likelihood)])
        # print('min measurement', min_meas)
        max_meas = np.max(measurement_likelihood)
        
        measure_matrix = np.array([[max_meas, 0], [0, 1]])

        # TODO: Parameterize the measurement likelihood as a Gaussian
        if measurement_likelihood is None:
            return
        maxids = np.unravel_index(measurement_likelihood.argmax(), measurement_likelihood.shape)
        d_max = self.d_min + (maxids[0] + 0.5) * self.delta_d
        phi_max = self.phi_min + (maxids[1] + 0.5) * self.delta_phi

        matrix_R = gaussian_filter(measure_matrix, sigma=0.6)
        print('this is the R noise filter shape', matrix_R.shape)

        # # find distance between centre of duckiebot to the center point of a segment
        # yellow_segments = [i for i in segmentsArray if i.color == 1]
        # # print('yellow  segments', yellow_segments)
        # x_yellow_1 = [i.points[0].x for i in segmentsArray if i.color == 1]
        # #print('this is x_yellow', x_yellow_1)
        # x_yellow_1_avg = np.mean(x_yellow_1)
        # # print('this is x1 average:', x_yellow_1_avg)
        #
        # x_yellow_2 = [i.points[1].x for i in segmentsArray if i.color == 1]
        # #print('this is x_yellow', x_yellow_2)
        # x_yellow_2_avg = np.mean(x_yellow_2)
        # # print('this is x2 average:', x_yellow_2_avg)
        #
        # x_c = int((x_yellow_1_avg + x_yellow_2_avg)/2)
        # #print('x_c : ', x_c)
        #
        # y_yellow_1 = [i.points[0].y for i in segmentsArray if i.color == 1]
        # #print('this is y_yellow', y_yellow_1)
        # y_yellow_1_avg = np.mean(y_yellow_1)
        # # print('this is y1 average:', y_yellow_1_avg)
        #
        # y_yellow_2 = [i.points[1].y for i in segmentsArray if i.color == 1]
        # #print('this is y_yellow', y_yellow_2)
        # y_yellow_2_avg = np.mean(y_yellow_2)
        # # print('this is y2 average:', y_yellow_2_avg)
        #
        # y_c = int((y_yellow_1_avg + y_yellow_2_avg)/2)
        # print('y_c : ', y_c)
        #
        # follow_point = x_c, y_c
        # seg_dist = int(np.sqrt(np.power(follow_point[0], 2) + np.power(follow_point[1], 2)))
        # #print('segment distance', seg_dist)
        #
        # y_comp_cov = int(y_c/seg_dist)
        # print('y comp of cov H', y_comp_cov)
        #
        # #u_t = np.array([[del_y, del_theta]])
        # angle_update = np.arctan(y_c/x_c)

        z_t = np.array([[d_max, phi_max]])
        print('this is z_t', z_t)
        print('this is the shape of', z_t.shape)
        # this is 2,1 matrix
        z_trans = z_t.T
        print('this is z transpose', z_trans.shape)
        #H = np.array([[y_comp_cov, 0], [0, angle_update]])
        H = np.array([[1, 0], [0, 1]])
        print('shape of h ', H.shape)
        H_gauss = gaussian_filter(H, sigma=0.5)
        print('this is h matrix', H_gauss)
        print('this is shape of H_gauss',H_gauss.shape)
        #pred_mu_up = np.array([self.belief['mean']])
        pred_mu_up = np.array(self.belief['mean'])
        print('this is pred mu in update', pred_mu_up)
        print('this is pred mu in update', pred_mu_up.shape)
        residual_mean = z_trans - H_gauss @ pred_mu_up
        print('this is residual mean', residual_mean)
        print('this is residual mean shape', residual_mean.shape)

        pred_cov_up = np.array(self.belief['covariance'])
        print('predicted_cov from pred in update step ', pred_cov_up)
        print('predicted_cov from pred in update step ', pred_cov_up.shape)
        #residual_cov = H_gauss @ pred_cov_up @ H_gauss.T + gaussian_param
        residual_cov = H_gauss @ pred_cov_up @ H_gauss.T + matrix_R
        print('this is the residual cov', residual_cov)
        print('this is the residual cov', residual_cov.shape)

        H_kal = np.array([[1, 0], [0, 1]])
        print('this is H_kal shape', H_kal.shape)

        # H_kal_gauss = (H_kal, sigma=0.5)
        #H_kal = np.array([[[h_multi_pdf, 0], [0, h_multi_pdf]]])
        kalman_gain = pred_cov_up @ H_kal @ np.linalg.inv(residual_cov)
        print('this is kalman gain', kalman_gain)
        print('this is kalman gain shape', kalman_gain.shape)

        updated_mu = pred_mu_up + kalman_gain @ residual_mean
        print('this is updated_mu', updated_mu)
        print('this is updated_mu shape', updated_mu.shape)
        updated_sigma = pred_cov_up - kalman_gain @ H_kal @ pred_cov_up
        print('this is updated_sigma', updated_sigma)
        print('this is updated_sigma', updated_sigma.shape)

        #self.belief = {'mean': updated_mu, 'covariance': updated_sigma}
        #print('this is belief in update', self.belief)

        # TODO: Apply the update equations for the Kalman Filter to self.belief


    def getEstimate(self):
        return self.belief

    def generate_measurement_likelihood(self, segments):

        if len(segments) == 0:
            return None

        grid = np.mgrid[self.d_min:self.d_max:self.delta_d,
                                    self.phi_min:self.phi_max:self.delta_phi]

        # initialize measurement likelihood to all zeros
        measurement_likelihood = np.zeros(grid[0].shape)

        for segment in segments:
            d_i, phi_i, l_i, weight = self.generateVote(segment)

            # if the vote lands outside of the histogram discard it
            if d_i > self.d_max or d_i < self.d_min or phi_i < self.phi_min or phi_i > self.phi_max:
                continue

            i = int(floor((d_i - self.d_min) / self.delta_d))
            j = int(floor((phi_i - self.phi_min) / self.delta_phi))
            measurement_likelihood[i, j] = measurement_likelihood[i, j] + 1

        if np.linalg.norm(measurement_likelihood) == 0:
            return None

        # lastly normalize so that we have a valid probability density function

        measurement_likelihood = measurement_likelihood / \
            np.sum(measurement_likelihood)
        return measurement_likelihood





    # generate a vote for one segment
    def generateVote(self, segment):
        p1 = np.array([segment.points[0].x, segment.points[0].y])
        p2 = np.array([segment.points[1].x, segment.points[1].y])
        t_hat = (p2 - p1) / np.linalg.norm(p2 - p1)

        n_hat = np.array([-t_hat[1], t_hat[0]])
        d1 = np.inner(n_hat, p1)
        d2 = np.inner(n_hat, p2)
        l1 = np.inner(t_hat, p1)
        l2 = np.inner(t_hat, p2)
        if (l1 < 0):
            l1 = -l1
        if (l2 < 0):
            l2 = -l2

        l_i = (l1 + l2) / 2
        d_i = (d1 + d2) / 2
        phi_i = np.arcsin(t_hat[1])
        if segment.color == segment.WHITE:  # right lane is white
            if(p1[0] > p2[0]):  # right edge of white lane
                d_i = d_i - self.linewidth_white
            else:  # left edge of white lane

                d_i = - d_i

                phi_i = -phi_i
            d_i = d_i - self.lanewidth / 2

        elif segment.color == segment.YELLOW:  # left lane is yellow
            if (p2[0] > p1[0]):  # left edge of yellow lane
                d_i = d_i - self.linewidth_yellow
                phi_i = -phi_i
            else:  # right edge of white lane
                d_i = -d_i
            d_i = self.lanewidth / 2 - d_i

        # weight = distance
        weight = 1
        return d_i, phi_i, l_i, weight

    def get_inlier_segments(self, segments, d_max, phi_max):
        inlier_segments = []
        for segment in segments:
            d_s, phi_s, l, w = self.generateVote(segment)
            if abs(d_s - d_max) < 3*self.delta_d and abs(phi_s - phi_max) < 3*self.delta_phi:
                inlier_segments.append(segment)
        return inlier_segments

    # get the distance from the center of the Duckiebot to the center point of a segment
    def getSegmentDistance(self, segment):
        x_c = (segment.points[0].x + segment.points[1].x) / 2
        y_c = (segment.points[0].y + segment.points[1].y) / 2
        return sqrt(x_c**2 + y_c**2)

    # prepare the segments for the creation of the belief arrays
    def prepareSegments(self, segments):
        segmentsArray = []
        self.filtered_segments = []
        for segment in segments:

            # we don't care about RED ones for now
            if segment.color != segment.WHITE and segment.color != segment.YELLOW:
                continue
            # filter out any segments that are behind us
            if segment.points[0].x < 0 or segment.points[1].x < 0:
                continue

            self.filtered_segments.append(segment)
            # only consider points in a certain range from the Duckiebot for the position estimation
            point_range = self.getSegmentDistance(segment)
            if point_range < self.range_est:
                segmentsArray.append(segment)

        return segmentsArray