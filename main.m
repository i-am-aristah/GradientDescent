% load dataset
ds = load("ex1data1.txt")

% split x/y
n = size(ds,2)-1;
x = ds(:,1:n);
y = ds(:,n+1);
m = length(y);

% normalise
[x, maxs, mins] = normalize(x, n);

% add column with ones - help hypothesis
xo = [ones(m,1),x];

% gradient descent
repeat = 1500;
lrate = 0.01;
thetas = zeros(n+1, 1);
[best, costs] = gradientdescent(repeat, lrate, thetas, xo, y, m, n);

% plot costs
plot(costs, 1:repeat);
% predict a value

p = [6;6;6];
pn = (p-maxs')./(maxs'-mins')
pn = [1;pn];
r = pn' * best
