import pickle

def extract_score(tree_path, index_path, min_size):
	index_dict = pickle.load(open(index_path, 'rb'))
	cluster_dict = dict()
	max_depth = 0
	with open(tree_path, 'r') as f:
		for line in f:
			clusters = line.split(' ')
			clusters[-1] = clusters[-1][:-1]
			if len(clusters) - 2 > max_depth:
				max_depth = len(clusters) - 2
			for x in range(2, len( clusters)):
				if int(clusters[x]) not in cluster_dict:
					if x == 2:
						cluster_dict[int(clusters[x])] = [(-1, 0), [index_dict[int(clusters[0])]]]
					else:
						cluster_dict[int(clusters[x])] = [(int(clusters[x-1]), x-2), [index_dict[int(clusters[0])]]]

				else:
					temp = cluster_dict[int(clusters[x])]
					temp[1].append(index_dict[int(clusters[0])])
					cluster_dict[int(clusters[x])] = temp

	level_score_dict = dict()
	for x in range(0, max_depth):
		level_score_dict[x] = [0, 0]

	for key in cluster_dict:
		cl = cluster_dict[key]
		underlying_dict = dict()
		max_count = [0, None]
		for file in cluster_dict[key][1]:
			category = int(file.split('_')[0])
			if category in underlying_dict:
				underlying_dict[category] += 1
			else:
				underlying_dict[category] = 1

			if underlying_dict[category] > max_count[0]:
				max_count = [underlying_dict[category], category]

		if len(cluster_dict[key][1]) >= min_size:
			temp = level_score_dict[cluster_dict[key][0][1]]
			temp[0] += max_count[0]
			temp[1] += len(cluster_dict[key][1])

	for x in range(0, max_depth):
		print('At level {}, the average accuracy is {}'.format(x, level_score_dict[x][0]/level_score_dict[x][1]))

	return level_score_dict
		




if __name__ == '__main__':
	extract_score('Test-Out/run002/mode.assign', 'order.pkl', 4)