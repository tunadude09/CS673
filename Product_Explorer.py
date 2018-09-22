import numpy as np

class Explorer(object):
    THRESHOLD = 10 # Calculated as follows: 1/1+e^-(THRESHOLD*distance-5)
                  # Higher number = less penalty for being close
    MIN_DISTANCE = 1 # Just to avoid division by 0, we need a minimum distance
    DISTANCE_EXPONENT = 2 # Higher makes it less likely to sample distant points
    DISTANCE_SCALER = 0.2 # Adjust this to make the scale of the graph make sense

    def __init__(self):
        self.kernels = 0

    #embedded_prods
    def generate_space(self, embedded_prods, percent_increase=0.2):
        self.known_points = embedded_prods
        xs = embedded_prods[:,0]
        ys = embedded_prods[:,1]
        minx = np.min(xs)
        miny = np.min(ys)
        maxx = np.max(xs)
        maxy = np.max(ys)
        self.min_x = minx - (maxx-minx)*percent_increase
        self.max_x = maxx + (maxx-minx)*percent_increase
        self.min_y = miny - (maxy-miny)*percent_increase
        self.max_y = maxy + (maxy-miny)*percent_increase

    def gridify_space(self, x_dim=20, y_dim=20):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.x_spacing = (self.max_x - self.min_x)/x_dim
        self.y_spacing = (self.max_y - self.min_y)/y_dim
        self.x_start = self.min_x + self.x_spacing/2
        self.y_start = self.min_y + self.y_spacing/2
        test_points = np.mgrid[self.x_start:self.x_start+self.x_dim*self.x_spacing:self.x_spacing, self.y_start:self.y_start+self.y_dim*self.y_spacing:self.y_spacing].reshape(2,-1).T
        return test_points

    def calc_min_dist(self, test_points):
        dists = np.empty((0,1))
        for i in xrange(test_points.shape[0]):
            mindist = np.min(np.sqrt(np.sum(np.power(self.known_points-test_points[i,:2], 2), axis=1)))
            dists = np.vstack((dists, np.atleast_2d(mindist)))
        return dists

    def distance_penalty(self, distance):
        penalty = 1/(1+np.exp(-(Explorer.THRESHOLD*Explorer.DISTANCE_SCALER*distance-5)))
        return penalty

    def pick_samples(self, evaluated_test_points, num_samples):
        largenumber = 9999999999 #this is used to offset picked samples so that they don't get picked again
        dists = self.calc_min_dist(evaluated_test_points)
        penalties = self.distance_penalty(dists)
        total = np.sum(evaluated_test_points[:,2] / ((Explorer.DISTANCE_SCALER*dists[:,0])**Explorer.DISTANCE_EXPONENT + Explorer.MIN_DISTANCE) * penalties[:,0])
        ws = np.random.uniform(0, total, num_samples)
        uptos = np.zeros(num_samples)
        samples = np.empty((0, 3))
        for i in xrange(evaluated_test_points.shape[0]):
            uptos[uptos<ws] += evaluated_test_points[i,2] / ((Explorer.DISTANCE_SCALER*dists[i,0])**Explorer.DISTANCE_EXPONENT + Explorer.MIN_DISTANCE) * penalties[i,0]
            for n in xrange(num_samples):
                if uptos[n] >= ws[n]:
                    samples = np.vstack((samples, evaluated_test_points[i]))
                    uptos[n] = -1 * total * largenumber
        return samples

    def evaluate_points_on_distribution_at_fixed_value(self, xs, value):
        ys = value / ((Explorer.DISTANCE_SCALER*xs)**Explorer.DISTANCE_EXPONENT + 1) * (self.distance_penalty(xs))
        return ys

    def sample_space(self, evaluated_test_points, num_samples=1):
        sample_boxes = self.pick_samples(evaluated_test_points, num_samples)
        print sample_boxes
        samples = np.copy(sample_boxes)
        samples = np.delete(samples, 2, 1)
        samples = np.delete(samples, 2, 1)
        sample_x_offsets = np.random.random(num_samples)
        sample_x_offsets *= self.x_spacing
        sample_x_offsets -= self.x_spacing/2
        sample_y_offsets = np.random.random(num_samples)
        sample_y_offsets *= self.y_spacing
        sample_y_offsets -= self.y_spacing/2
        samples[:,0] += sample_x_offsets
        samples[:,1] += sample_y_offsets
        return samples
