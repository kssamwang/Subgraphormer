import torch
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score
import numpy as np
import matplotlib.pyplot as plt

def rank_data(data, k):
	# 计算排序后的分位数切点
	quantiles = np.linspace(0, 1, k + 1)
	
	# 根据切点得到数据的分位段
	sorted_indices = np.argsort(data)
	sorted_data = data[sorted_indices]
	
	# 得到每个段的阈值
	thresholds = np.quantile(sorted_data, quantiles)
	
	# 初始化结果数组
	rankdata = np.zeros(len(data), dtype=int)
	
	# 根据阈值划分数据段
	for i in range(k):
		if i == k - 1:
			rankdata[sorted_data >= thresholds[i]] = i
		else:
			rankdata[(sorted_data >= thresholds[i]) & (sorted_data < thresholds[i + 1])] = i
	
	# 恢复原顺序
	rankdata = rankdata[np.argsort(sorted_indices)]
	
	return rankdata

def _eval_f1(y, pred, average, attribute):
	f1_scores = []
	
	# return asc sorted attr_values
	attr_values, indices = np.unique(attribute, return_inverse=True)
	
	# calculate accuracy / f1 by attribute
	for idx, _ in enumerate(attr_values):
		mask = indices == idx
		y_subset = y[mask]
		pred_subset = pred[mask]
		f1 = f1_score(y_subset, pred_subset, average=average)
		f1_scores.append(f1)
	
	return attr_values, np.array(f1_scores)

def _eval_acc(y, pred, attribute):
	acc_scores = []
	
	# return asc sorted attr_values
	attr_values, indices = np.unique(attribute, return_inverse=True)
	
	# calculate accuracy / acc by attribute
	for idx, _ in enumerate(attr_values):
		mask = indices == idx
		y_subset = y[mask]
		pred_subset = pred[mask]
		acc = accuracy_score(y_subset, pred_subset)
		acc_scores.append(acc)
	
	return attr_values, np.array(acc_scores)

def _rmse(y_true,y_pred):
	rmse_list = []
	for i in range(y_true.shape[1]):
		# ignore nan values
		is_labeled = y_true[:,i] == y_true[:,i]
		rmse_list.append(np.sqrt(((y_true[is_labeled,i] - y_pred[is_labeled,i])**2).mean()))
	return sum(rmse_list)/len(rmse_list)

def _eval_rmse(y, pred, attribute):
	rmse_scores = []
	
	# return asc sorted attr_values
	attr_values, indices = np.unique(attribute, return_inverse=True)
	
	# calculate accuracy / rmse by attribute
	for idx, _ in enumerate(attr_values):
		mask = indices == idx
		y_subset = y[mask]
		pred_subset = pred[mask]
		rmse = _rmse(y_subset, pred_subset)
		rmse_scores.append(rmse)
	
	return attr_values, np.array(rmse_scores)

def _rocauc(y_true, y_pred):
	rocauc_list = []
	for i in range(y_true.shape[1]):
		#AUC is only defined when there is at least one positive data.
		if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
			# ignore nan values
			is_labeled = y_true[:,i] == y_true[:,i]
			rocauc_list.append(roc_auc_score(y_true[is_labeled,i], y_pred[is_labeled,i]))
	if len(rocauc_list) == 0:
		return 0.0

	return sum(rocauc_list)/len(rocauc_list)

def _eval_rocauc(y, pred, attribute):
	rocauc_scores = []
	
	# return asc sorted attr_values
	attr_values, indices = np.unique(attribute, return_inverse=True)
	
	# calculate accuracy / rocauc by attribute
	for idx, _ in enumerate(attr_values):
		mask = indices == idx
		y_subset = y[mask]
		pred_subset = pred[mask]
		rocauc = _rocauc(y_subset, pred_subset)
		rocauc_scores.append(rocauc)
	
	return attr_values, np.array(rocauc_scores)

def _ap(y_true, y_pred):
	ap_list = []
	for i in range(y_true.shape[1]):
		#AUC is only defined when there is at least one positive data.
		if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
			# ignore nan values
			is_labeled = y_true[:,i] == y_true[:,i]
			ap = average_precision_score(y_true[is_labeled,i], y_pred[is_labeled,i])
			ap_list.append(ap)
	if len(ap_list) == 0:
		return 0.0
	return sum(ap_list)/len(ap_list)

def _eval_ap(y, pred, attribute):
	ap_scores = []
	
	# return asc sorted attr_values
	attr_values, indices = np.unique(attribute, return_inverse=True)
	
	# calculate accuracy / AP by attribute
	for idx, _ in enumerate(attr_values):
		mask = indices == idx
		y_subset = y[mask]
		pred_subset = pred[mask]
		ap = _ap(y_subset, pred_subset)
		ap_scores.append(ap)

	return attr_values, np.array(ap_scores)

def compute_metrics_by_attribute(eval_metric, y, pred, attribute):
	if eval_metric == 'rocauc':
		attr_values, metrics = _eval_rocauc(y, pred, attribute)
	elif eval_metric == 'ap':
		attr_values, metrics = _eval_ap(y, pred, attribute)
	elif eval_metric == 'rmse':
		attr_values, metrics = _eval_rmse(y, pred, attribute)
	elif eval_metric == 'acc':
		attr_values, metrics = _eval_acc(y, pred, attribute)
	elif eval_metric.endswith('f1'):
		average = eval_metric[:-3]
		avarage_kws = ['micro', 'macro', 'samples', 'weighted', 'binary']
		assert(average in avarage_kws)
		attr_values, metrics = _eval_f1(y, pred, average, attribute)
	else:
		raise ValueError('Undefined eval metric %s ' % (eval_metric))

	return attr_values, metrics

def plot_bar_chart(x, y, metric, filename):
	plt.figure(figsize=(10, 6))
	plt.bar(x, y, color='lightsteelblue', edgecolor='black')
	plt.xlim(min(x) - (max(x) - min(x)) * 0.1, max(x) + (max(x) - min(x)) * 0.1)
	plt.ylim(0, max(y) + max(y) * 0.1)
	plt.title('%s by Graph Size' % metric)
	plt.xlabel('graph size')
	plt.ylabel(metric)
	plt.savefig(filename)

if __name__ == '__main__':
	N = 100
	num_classes = 10

	y = torch.randint(0, 2, (N, num_classes))
	pred = torch.randint(0, 2, (N, num_classes))

	V = torch.randint(1, 101, (N,)).numpy()
	E = torch.randint(1, 101, (N,)).numpy()

	# 如果用rank_data处理，即从按照每个size值变成按照每个size的k分段
	k = 5
	V = rank_data(V,k)
	E = rank_data(E,k)

	asc_V, f1_v = compute_metrics_by_attribute("micro-f1", y.numpy(), pred.numpy(), V)
	asc_E, f1_e = compute_metrics_by_attribute("micro-f1", y.numpy(), pred.numpy(), E)

	asc_V_rocauc, rocauc_v = compute_metrics_by_attribute("rocauc", y.numpy(), pred.numpy(), V)
	asc_E_rocauc, rocauc_e = compute_metrics_by_attribute("rocauc", y.numpy(), pred.numpy(), E)

	asc_V_ap, ap_v = compute_metrics_by_attribute("ap", y.numpy(), pred.numpy(), V)
	asc_E_ap, ap_e = compute_metrics_by_attribute("ap", y.numpy(), pred.numpy(), E)

	asc_V_rmse, rmse_v = compute_metrics_by_attribute("rmse", y.numpy(), pred.numpy(), V)
	asc_E_rmse, rmse_e = compute_metrics_by_attribute("rmse", y.numpy(), pred.numpy(), E)

	# 打印结果以验证正确性
	print("F1 scores by V:", f1_v)
	print("F1 scores by E:", f1_e)
	print("ROC-AUC by V:", rocauc_v)
	print("ROC-AUC by E:", rocauc_e)
	print("AP by V:", ap_v)
	print("AP by E:", ap_e)
	print("RMSE by V:", rmse_v)
	print("RMSE by E:", rmse_e)

	# 生成图表
	plot_bar_chart(asc_V, f1_v, "micro-f1", "f1_v.png")
	plot_bar_chart(asc_E, f1_e, "micro-f1", "f1_e.png")
	plot_bar_chart(asc_V_rocauc, rocauc_v, "rocauc", "rocauc_v.png")
	plot_bar_chart(asc_E_rocauc, rocauc_e, "rocauc", "rocauc_e.png")
	plot_bar_chart(asc_V_ap, ap_v, "AP", "ap_v.png")
	plot_bar_chart(asc_E_ap, ap_e, "AP", "ap_e.png")
	plot_bar_chart(asc_V_rmse, rmse_v, "RMSE", "rmse_v.png")
	plot_bar_chart(asc_E_rmse, rmse_e, "RMSE", "rmse_e.png")

	# 对于accuracy测试，将y和pred设为一维
	y = torch.randint(0, num_classes, (N,))
	pred = torch.randint(0, num_classes, (N,))

	# 计算其他指标
	asc_V_acc, acc_v = compute_metrics_by_attribute("acc", y.numpy(), pred.numpy(), V)
	asc_E_acc, acc_e = compute_metrics_by_attribute("acc", y.numpy(), pred.numpy(), E)

	# 打印结果以验证正确性
	print("Accuracy by V:", acc_v)
	print("Accuracy by E:", acc_e)

	plot_bar_chart(asc_V_acc, acc_v, "Accuracy", "acc_v.png")
	plot_bar_chart(asc_E_acc, acc_e, "Accuracy", "acc_e.png")