data  = csvread('train_X.csv');
label = csvread('train_Y.csv');
data  = [data, label];

K = max(label);
data_sampled = [];
lim = 175;
for ii = 0:K-1
	idx = (label == ii);
	temp = data(find(idx),:);
	nk = sum(idx);
	if nk > lim
		ridx = randperm(nk);
		temp2 = temp(ridx(1:lim),:);
	else
		temp2 = temp;
	end
	data_sampled = [data_sampled; temp2];
end
shuffled_data = data_sampled(randperm(size(data_sampled,1)),:);
csvwrite('data_sampled.csv', shuffled_data);
