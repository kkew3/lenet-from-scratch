#include "ex5_10.h"

int main()
{
	std::random_device rd;
	std::mt19937 gen(rd());
	MNIST mnist;
	mnist.read_from_file("ex5_10assets");
	std::vector<MNIST::DImg> norm_X_train, norm_X_test;
	normalize(mnist, norm_X_train, norm_X_test);
	NormalizedMNIST norm_mnist(norm_X_train, mnist.y_train, norm_X_test,
			mnist.y_test);
	LeNetParams params;
	params.init(gen);
	TrainRecord rep;
	train(5, 0.01, gen, params, norm_mnist, rep);
	rep.write_to_file("ex5_10assets/loss_acc.txt");

	return 0;
}

double dot(const Vec &u, const Vec &v)
{
	assert(u.shape(0) == v.shape(0));
	double x = 0.0;
	for (size_t k = 0; k < u.shape(0); ++k)
	{
		x += u({ k }) * v({ k });
	}
	return x;
}

// Tested
Mat matmul(const Mat &u, const Mat &v)
{
	assert(u.shape(1) == v.shape(0));
	Mat z({ u.shape(0), v.shape(1) });
	const size_t M = z.shape(0), N = z.shape(1), P = u.shape(1);
	for (size_t i = 0; i < M; ++i)
	{
		for (size_t j = 0; j < N; ++j)
		{
			double x = 0.0;
			for (size_t k = 0; k < P; ++k)
			{
				x += u({ i, k }) * v({ k, j });
			}
			z({ i, j }) = x;
		}
	}
	return z;
}


void mvadd_(Mat &u, const Vec &v)
{
	assert(u.shape(0) == v.shape(0));
	const size_t M = u.shape(0), N = u.shape(1);
	for (size_t i = 0; i < M; ++i)
	{
		for (size_t j = 0; j < N; ++j)
		{
			u({ i, j }) += v({ i });
		}
	}
}

void mvadd2_(Mat &u, const Vec &v)
{
	assert(u.shape(1) == v.shape(0));
	const size_t M = u.shape(0), N = u.shape(1);
	for (size_t i = 0; i < M; ++i)
	{
		for (size_t j = 0; j < N; ++j)
		{
			u({ i, j }) += v({ j });
		}
	}
}

// Tested
Mat htile(const Mat &u, size_t n)
{
	Mat z({ u.shape(0), n * u.shape(1) });
	for (size_t i = 0; i < u.shape(0); ++i)
	{
		for (size_t j = 0; j < u.shape(1); ++j)
		{
			for (size_t k = 0; k < n; ++k)
			{
				z({ i, j + k * u.shape(1) }) = u({ i, j });
			}
		}
	}
	return z;
}

// Tested
Tsr4 conv2d_valid(const Tsr4 &X, const Tsr4 &u, const Vec &b)
{
	const size_t N = X.shape(0), C = X.shape(1), H = X.shape(2), W = X.shape(3);
	const size_t D = u.shape(0), K = u.shape(2);
	assert(u.shape(1) == C);
	assert(u.shape(3) == K);
	assert(K <= H && K <= W);
	const size_t oH = H - K + 1, oW = W - K + 1;
	// D row vectors to be dot producted with input vectors. v_u is of shape
	// (D, K*K*C)
	Mat v_u = Mat(std::move(u.data()), { D, C *K * K });
	// a mini-batch of inputs produce this number of vectors to be dot producted
	// with filter weights
	const size_t M = oH * oW * N;
	// to be M column vectors of inputs
	Mat v_X({ v_u.shape(1), M });
	// populate `v_X`
	for (size_t n = 0; n < N; ++n)
	{
		for (size_t i = 0; i < oH; ++i)
		{
			for (size_t j = 0; j < oW; ++j)
			{
				for (size_t k = 0; k < C; ++k)
				{
					for (size_t ii = 0; ii < K; ++ii)
					{
						for (size_t jj = 0; jj < K; ++jj)
						{
							v_X({ indexcont3({ k, ii, jj }, { C, K, K }),
											indexcont3({ n, i, j }, { N, oH, oW }) })
								= X({ n, k, i + ii, j + jj });
						}
					}
				}
			}
		}
	}
	// the result in vector form
	Mat v_r = matmul(v_u, v_X);
	mvadd_(v_r, b);

	// to be the final result
	Tsr4 r({ N, D, oH, oW });
	// populate `r`
	for (size_t n = 0; n < N; ++n)
	{
		for (size_t k = 0; k < D; ++k)
		{
			for (size_t i = 0; i < oH; ++i)
			{
				for (size_t j = 0; j < oW; ++j)
				{
					r({ n, k, i, j }) = v_r({ k, indexcont3({ n, i, j }, { N, oH, oW }) });
				}
			}
		}
	}
	return r;
}

Tsr4 pad2d(const Tsr4 &u, size_t p)
{
	const size_t N = u.shape(0), C = u.shape(1), H = u.shape(2), W = u.shape(3);
	const size_t oH = H + 2 * p, oW = W + 2 * p;
	Tsr4 r({ N, C, oH, oW });
	for (size_t n = 0; n < N; ++n)
	{
		for (size_t k = 0; k < C; ++k)
		{
			for (size_t i = 0; i < oH; ++i)
			{
				for (size_t j = 0; j < oW; ++j)
				{
					if (i < p || i >= oH - p || j < p || j >= oW - p)
					{
						r({ n, k, i, j }) = 0.0;
					}
					else
					{
						r({ n, k, i, j }) = u({ n, k, i - p, j - p });
					}
				}
			}
		}
	}
	return r;
}

// Tested
std::pair<Tsr4, Tsr4> maxpool2d(const Tsr4 &u, size_t p)
{
	const size_t N = u.shape(0), C = u.shape(1), H = u.shape(2), W = u.shape(3);
	assert(H >= p && W >= p);
	assert(H % p == 0 && W % p == 0);
	const size_t oH = H / p, oW = W / p;
	const size_t p2 = p * p;
	// the result tensor
	Tsr4 r({ N, C, oH, oW });
	// the mask tensor
	Tsr4 m({ N, C, H, W });
	for (size_t n = 0; n < N; ++n)
	{
		for (size_t k = 0; k < C; ++k)
		{
			for (size_t i = 0; i < H; i += p)
			{
				for (size_t j = 0; j < W; j += p)
				{
					std::vector<double> f(p2);
					for (size_t ii = i; ii < i + p; ++ii)
					{
						for (size_t jj = j; jj < j + p; ++jj)
						{
							f[indexcont2({ ii - i, jj - j }, { p, p })] = u({ n, k, ii, jj });
						}
					}
					auto argmax_ptr = std::max_element(f.begin(), f.end());
					r({ n, k, i / p, j / p }) = *argmax_ptr;
					size_t argmax_idx = std::distance(f.begin(), argmax_ptr);
					for (size_t ii = i; ii < i + p; ++ii)
					{
						for (size_t jj = j; jj < j + p; ++jj)
						{
							m({ n, k, ii, jj }) = indexcont2({ ii - i, jj - j }, { p, p }) == argmax_idx
									? 1.0
									: 0.0;
						}
					}
				}
			}
		}
	}
	return std::make_pair(std::move(r), std::move(m));
}

Mat mflatten(const Tsr4 &u)
{
	const size_t N = u.shape(0), C = u.shape(1), H = u.shape(2), W = u.shape(3);
	const size_t oC = C * H * W;
	Mat z(std::move(u.data()), { N, oC });
	return z;
}

void relu_(Tsr4 &u)
{
	for (size_t n = 0; n < u.shape(0); ++n)
	{
		for (size_t k = 0; k < u.shape(1); ++k)
		{
			for (size_t i = 0; i < u.shape(2); ++i)
			{
				for (size_t j = 0; j < u.shape(3); ++j)
				{
					u({ n, k, i, j }) = std::max(0.0, u({ n, k, i, j }));
				}
			}
		}
	}
}

void relu_(Mat &u)
{
	for (size_t n = 0; n < u.shape(0); ++n)
	{
		for (size_t k = 0; k < u.shape(1); ++k)
		{
			u({ n, k }) = std::max(0.0, u({ n, k }));
		}
	}
}

Mat linear(const Mat &X, const Mat &w, const Vec &b)
{
	Mat z = matmul(X, w);
	mvadd2_(z, b);
	return z;
}

// Tested
Mat softmax(const Mat &X)
{
	Mat z({ X.shape(0), X.shape(1) });
	for (size_t i = 0; i < X.shape(0); ++i)
	{
		// stable softmax
		// Reference: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
		std::vector<double> row(X.shape(1));
		for (size_t j = 0; j < X.shape(1); ++j)
		{
			row[j] = X({ i, j });
		}
		double max = *std::max_element(row.begin(), row.end());
		double sum = 0.0;
		for (size_t j = 0; j < X.shape(1); ++j)
		{
			z({ i, j }) = std::exp(X({ i, j }) - max);
			sum += z({ i, j });
		}
		for (size_t j = 0; j < X.shape(1); ++j)
		{
			z({ i, j }) /= sum;
		}
	}
	return z;
}

double celoss(const Mat &z, const LabelVec &t)
{
	const size_t N = z.shape(0);
	double l = 0.0;
	for (size_t n = 0; n < N; ++n)
	{
		l -= std::log(z({ n, t({ n }) }));
	}
	l /= static_cast<double>(N);
	return l;
}

Mat b_celoss(const Mat &z, const LabelVec &t)
{
	const size_t N = z.shape(0), C = z.shape(1);
	Mat dz({ N, C });
	for (size_t n = 0; n < N; ++n)
	{
		for (size_t k = 0; k < C; ++k)
		{
			if (k == t({ n }))
			{
				dz({ n, k }) = -1.0 / (static_cast<double>(N) * z({ n, k }));
			}
			else
			{
				dz({ n, k }) = 0.0;
			}
		}
	}
	return dz;
}

Mat b_softmax(const Mat &da, const Mat &a, const LabelVec &t)
{
	const size_t N = a.shape(0), C = a.shape(1);
	Mat dz({ N, C });
	for (size_t n = 0; n < N; ++n)
	{
		const size_t tn = t({ n });
		for (size_t k = 0; k < C; ++k)
		{
			dz({ n, k }) = da({ n, tn });
			if (k == tn)
			{
				dz({ n, k }) *= a({ n, tn }) * (1 - a({ n, tn }));
			}
			else
			{
				dz({ n, k }) *= -a({ n, tn }) * a({ n, k });
			}
		}
	}
	return dz;
}

std::tuple<Mat, Mat, Vec> b_linear(const Mat &da, const Mat &z, const Mat &w)
{
	const size_t N = z.shape(0), C = z.shape(1), D = da.shape(1);
	Mat dz({ N, C });
	for (size_t n = 0; n < N; ++n)
	{
		for (size_t i = 0; i < C; ++i)
		{
			dz({ n, i }) = 0.0;
			for (size_t j = 0; j < D; ++j)
			{
				dz({ n, i }) += da({ n, j }) * w({ i, j });
			}
		}
	}
	Mat dw({ C, D });
	for (size_t i = 0; i < C; ++i)
	{
		for (size_t j = 0; j < D; ++j)
		{
			dw({ i, j }) = 0.0;
			for (size_t n = 0; n < N; ++n)
			{
				dw({ i, j }) += da({ n, j }) * z({ n, i });
			}
		}
	}
	Vec db({ D });
	for (size_t j = 0; j < D; ++j)
	{
		db({ j }) = 0.0;
		for (size_t n = 0; n < N; ++n)
		{
			db({ j }) += da({ n, j });
		}
	}
	return std::make_tuple(std::move(dz), std::move(dw), std::move(db));
}

Mat b_relu(const Mat &da, const Mat &a)
{
	const size_t N = da.shape(0), C = da.shape(1);
	Mat dz({ N, C });
	for (size_t i = 0; i < N; ++i)
	{
		for (size_t j = 0; j < C; ++j)
		{
			dz({ i, j }) = a({ i, j }) > 0.0 ? da({ i, j }) : 0.0;
		}
	}
	return dz;
}

Tsr4 b_relu(const Tsr4 &da, const Tsr4 &a)
{
	const size_t N = da.shape(0), C = da.shape(1), H = da.shape(2),
				 W = da.shape(3);
	Tsr4 dz({ N, C, H, W });
	for (size_t n = 0; n < N; ++n)
	{
		for (size_t k = 0; k < C; ++k)
		{
			for (size_t i = 0; i < H; ++i)
			{
				for (size_t j = 0; j < W; ++j)
				{
					dz({ n, k, i, j }) = a({ n, k, i, j }) > 0.0 ? da({ n, k, i, j }) : 0.0;
				}
			}
		}
	}
	return dz;
}

Tsr4 b_mflatten(const Mat &u, const std::array<size_t, 4> &shape)
{
	return Tsr4(std::move(u.data()), shape);
}

Tsr4 b_pad2d(const Tsr4 &da, size_t p)
{
	const size_t N = da.shape(0), C = da.shape(1), H = da.shape(2),
				 W = da.shape(3);
	assert(static_cast<int>(H) - 2 * static_cast<int>(p) > 0);
	assert(static_cast<int>(W) - 2 * static_cast<int>(p) > 0);
	const size_t oH = H - 2 * p, oW = W - 2 * p;
	Tsr4 dz({ N, C, oH, oW });
	for (size_t n = 0; n < N; ++n)
	{
		for (size_t k = 0; k < C; ++k)
		{
			for (size_t i = 0; i < oH; ++i)
			{
				for (size_t j = 0; j < oW; ++j)
				{
					dz({ n, k, i, j }) = da({ n, k, i + p, j + p });
				}
			}
		}
	}
	return dz;
}

Tsr4 b_maxpool2d(const Tsr4 &da, const Tsr4 &m, size_t p)
{
	const size_t N = m.shape(0), C = m.shape(1), oH = m.shape(2), oW = m.shape(3);
	const size_t H = da.shape(2), W = da.shape(3);
	assert(da.shape(0) == N && da.shape(1) == C);
	assert(oH == H * p && oW == W * p);
	Tsr4 dz({ N, C, oH, oW });
	for (size_t n = 0; n < N; ++n)
	{
		for (size_t k = 0; k < C; ++k)
		{
			for (size_t i = 0; i < oH; ++i)
			{
				for (size_t j = 0; j < oW; ++j)
				{
					const size_t ii = i / p, jj = j / p;
					dz({ n, k, i, j }) = m({ n, k, i, j }) * da({ n, k, ii, jj });
				}
			}
		}
	}
	return dz;
}

std::tuple<Tsr4, Tsr4, Vec> b_conv2d_valid(const Tsr4 &da, const Tsr4 &z,
		const Tsr4 &w)
{
	const size_t N = da.shape(0), D = da.shape(1), oH = da.shape(2),
				 oW = da.shape(3);
	const size_t C = z.shape(1), H = z.shape(2), W = z.shape(3);
	const size_t K = w.shape(2);
	assert(z.shape(0) == N);
	assert(w.shape(0) == D && w.shape(1) == C && w.shape(3) == K);
	Tsr4 dz({ N, C, H, W });
	for (size_t n = 0; n < N; ++n)
	{
		for (size_t c = 0; c < C; ++c)
		{
			for (size_t i = 0; i < H; ++i)
			{
				for (size_t j = 0; j < W; ++j)
				{
					double r = 0.0;
					for (size_t d = 0; d < D; ++d)
					{
						for (size_t ii = 0; ii < oH; ++ii)
						{
							for (size_t jj = 0; jj < oW; ++jj)
							{
								if (i >= ii && i <= ii + K - 1 && j >= jj && j <= jj + K - 1)
								{
									r += da({ n, d, ii, jj }) * w({ d, c, i - ii, j - jj });
								}
							}
						}
					}
					dz({ n, c, i, j }) = r;
				}
			}
		}
	}
	Tsr4 dw({ D, C, K, K });
	for (size_t d = 0; d < D; ++d)
	{
		for (size_t c = 0; c < C; ++c)
		{
			for (size_t ki = 0; ki < K; ++ki)
			{
				for (size_t kj = 0; kj < K; ++kj)
				{
					double r = 0.0;
					for (size_t n = 0; n < N; ++n)
					{
						for (size_t ii = 0; ii < oH; ++ii)
						{
							for (size_t jj = 0; jj < oW; ++jj)
							{
								r += da({ n, d, ii, jj }) * z({ n, c, ii + ki, jj + kj });
							}
						}
					}
					dw({ d, c, ki, kj }) = r;
				}
			}
		}
	}
	Vec db({ D });
	for (size_t d = 0; d < D; ++d)
	{
		double r = 0.0;
		for (size_t n = 0; n < N; ++n)
		{
			for (size_t ii = 0; ii < oH; ++ii)
			{
				for (size_t jj = 0; jj < oW; ++jj)
				{
					r += da({ n, d, ii, jj });
				}
			}
		}
		db({ d }) = r;
	}

	return std::make_tuple(dz, dw, db);
}

void tsrsub_(Tsr4 &u, const Tsr4 &v, double scale)
{
	assert(u.shape(0) == v.shape(0) && u.shape(1) == v.shape(1)
			&& u.shape(2) == v.shape(2) && u.shape(3) == v.shape(3));
	const size_t M = u.shape(0), N = u.shape(1), P = u.shape(2), Q = u.shape(3);
	for (size_t i = 0; i < M; ++i)
	{
		for (size_t j = 0; j < N; ++j)
		{
			for (size_t k = 0; k < P; ++k)
			{
				for (size_t l = 0; l < Q; ++l)
				{
					u({ i, j, k, l }) -= scale * v({ i, j, k, l });
				}
			}
		}
	}
}

void tsrsub_(Mat &u, const Mat &v, double scale)
{
	assert(u.shape(0) == v.shape(0) && u.shape(1) == v.shape(1));
	const size_t M = u.shape(0), N = u.shape(1);
	for (size_t i = 0; i < M; ++i)
	{
		for (size_t j = 0; j < N; ++j)
		{
			u({ i, j }) -= scale * v({ i, j });
		}
	}
}

void tsrsub_(Vec &u, const Vec &v, double scale)
{
	assert(u.shape(0) == v.shape(0));
	const size_t M = u.shape(0);
	for (size_t i = 0; i < M; ++i)
	{
		u({ i }) -= scale * v({ i });
	}
}

LabelVec margmax(const Mat &a)
{
	std::vector<size_t> prediction;
	const size_t N = a.shape(0), C = a.shape(1);
	for (size_t i = 0; i < N; ++i)
	{
		std::vector<double> row(C);
		for (size_t j = 0; j < C; ++j)
		{
			row[j] = a({i, j});
		}
		prediction.push_back(std::distance(row.begin(), std::max_element(row.begin(),
								row.end())));
	}
	return LabelVec(std::move(prediction), {N});
}

size_t n_matches(const LabelVec &y, const LabelVec &t)
{
	assert(y.shape(0) == t.shape(0));
	size_t n_same = 0;
	const size_t N = y.shape(0);
	for (size_t i = 0; i < N; ++i)
	{
		if (y({i}) == t({i}))
		{
			++n_same;
		}
	}
	return n_same;
}

std::pair<Mat, double> lenet(const Tsr4 &z, LeNetParams &params,
		const LabelVec &t, bool to_grad)
{
	assert(z.shape(1) == 1 && z.shape(2) == 28 && z.shape(3) == 28);
	Tsr4 z1 = pad2d(z, 2);
	Tsr4 z2 = conv2d_valid(z1, params.conv1_w, params.conv1_b);
	assert(z2.shape(1) == 6 && z2.shape(2) == 28 && z2.shape(3) == 28);
	relu_(z2);
	auto pair_z3_m3 = maxpool2d(z2, 2);
	assert(pair_z3_m3.first.shape(1) == 6 && pair_z3_m3.first.shape(2) == 14
			&& pair_z3_m3.first.shape(3) == 14);
	Tsr4 z4 = pad2d(pair_z3_m3.first, 2);
	Tsr4 z5 = conv2d_valid(z4, params.conv2_w, params.conv2_b);
	assert(z5.shape(1) == 16 && z5.shape(2) == 14 && z5.shape(3) == 14);
	relu_(z5);
	auto pair_z6_m6 = maxpool2d(z5, 2);
	assert(pair_z6_m6.first.shape(1) == 16 && pair_z6_m6.first.shape(2) == 7
			&& pair_z6_m6.first.shape(3) == 7);
	Mat z7 = mflatten(pair_z6_m6.first);
	assert(z7.shape(1) == 784);
	Mat z8 = linear(z7, params.linear1_w, params.linear1_b);
	relu_(z8);
	Mat z9 = linear(z8, params.linear2_w, params.linear2_b);
	relu_(z9);
	Mat z10 = linear(z9, params.linear3_w, params.linear3_b);
	Mat z11 = softmax(z10);
	double loss = celoss(z11, t);

	if (to_grad)
	{
		Mat dz11 = b_celoss(z11, t);
		Mat dz10 = b_softmax(dz11, z11, t);
		auto tup_dz9_dw_db = b_linear(dz10, z9, params.linear3_w);
		params.grad_linear3_w = std::get<1>(tup_dz9_dw_db);
		params.grad_linear3_b = std::get<2>(tup_dz9_dw_db);
		Mat dz9_ = b_relu(std::get<0>(tup_dz9_dw_db), z9); // dz9 before relu
		auto tup_dz8_dw_db = b_linear(dz9_, z8, params.linear2_w);
		params.grad_linear2_w = std::get<1>(tup_dz8_dw_db);
		params.grad_linear2_b = std::get<2>(tup_dz8_dw_db);
		Mat dz8_ = b_relu(std::get<0>(tup_dz8_dw_db), z8); // dz8 before relu
		auto tup_dz7_dw_db = b_linear(dz8_, z7, params.linear1_w);
		params.grad_linear1_w = std::get<1>(tup_dz7_dw_db);
		params.grad_linear1_b = std::get<2>(tup_dz7_dw_db);
		Tsr4 dz6 = b_mflatten(std::get<0>(tup_dz7_dw_db), { z.shape(0), 16, 7, 7 });
		Tsr4 dz5 = b_maxpool2d(dz6, pair_z6_m6.second, 2);
		Tsr4 dz5_ = b_relu(dz5, z5); // dz5 before relu
		auto tup_dz4_dw_db = b_conv2d_valid(dz5_, z4, params.conv2_w);
		params.grad_conv2_w = std::get<1>(tup_dz4_dw_db);
		params.grad_conv2_b = std::get<2>(tup_dz4_dw_db);
		Tsr4 dz3 = b_pad2d(std::get<0>(tup_dz4_dw_db), 2);
		Tsr4 dz2 = b_maxpool2d(dz3, pair_z3_m3.second, 2);
		Tsr4 dz2_ = b_relu(dz2, z2); // dz2 before relu
		auto tup_dz1_dw_db = b_conv2d_valid(dz2_, z1, params.conv1_w);
		params.grad_conv1_w = std::get<1>(tup_dz1_dw_db);
		params.grad_conv1_b = std::get<2>(tup_dz1_dw_db);
	}

	return std::make_pair(z11, loss);
}

bool MNIST::_read_y_train(const std::string &basedir)
{
	if (basedir.back() == '/')
	{
		throw std::invalid_argument("basedir");
	}
	std::string filename(basedir);
	filename.append("/train-labels-idx1-ubyte");
	std::ifstream infile(filename, std::ios::binary);
	U_MNISTLabelFileHeader u_header;
	infile.read(u_header.bytes, sizeof(U_MNISTLabelFileHeader));
	if (infile.gcount() < sizeof(U_MNISTLabelFileHeader))
	{
		std::cerr << "read_mnist: error: failed to read bytes\n";
		return false;
	}
	unsigned int magic = ntohl(u_header.h.magic);
	unsigned int n = ntohl(u_header.h.n);
	if (magic != 2049)
	{
		std::cerr << "read_mnist: error: magic number mismatch\n";
		return false;
	}
	if (n != 60000)
	{
		std::cerr << "read_mnist: error: number of items mismatch\n";
		return false;
	}
	std::istreambuf_iterator<char> begin(infile), end;
	std::vector<_LabelType> tmp(begin, end);
	if (tmp.size() != n)
	{
		std::cerr << "read_mnist: error: number of labels mismatch\n";
		return false;
	}
	y_train.swap(tmp);
	assert(y_train.size() == n);
	return true;
}

bool MNIST::_read_X_train(const std::string &basedir)
{
	if (basedir.back() == '/')
	{
		throw std::invalid_argument("basedir");
	}
	std::string filename(basedir);
	filename.append("/train-images-idx3-ubyte");
	std::ifstream infile(filename, std::ios::binary);
	U_MNISTImageFileHeader u_header;
	infile.read(u_header.bytes, sizeof(U_MNISTImageFileHeader));
	if (infile.gcount() < sizeof(U_MNISTImageFileHeader))
	{
		std::cerr << "read_mnist: error: failed to read bytes\n";
		return false;
	}
	unsigned int magic = ntohl(u_header.h.magic);
	unsigned int n = ntohl(u_header.h.n);
	unsigned int n_rows = ntohl(u_header.h.n_rows);
	unsigned int n_cols = ntohl(u_header.h.n_cols);
	if (magic != 2051)
	{
		std::cerr << "read_mnist: error: magic number mismatch\n";
		return false;
	}
	if (n != 60000)
	{
		std::cerr << "read_mnist: error: number of items mismatch\n";
		return false;
	}
	if (n_rows != 28)
	{
		std::cerr << "read_mnist: error: number of rows mismatch\n";
		return false;
	}
	if (n_cols != 28)
	{
		std::cerr << "read_mnist: error: number of columns mismatch\n";
		return false;
	}
	std::istreambuf_iterator<char> begin(infile), end;
	std::vector<_PixelType> tmp(begin, end);
	if (tmp.size() != n * n_rows * n_cols)
	{
		std::cerr << "read_mnist: error: number of pixels mismatch\n";
		return false;
	}

	auto imgbegin = tmp.begin();
	auto imgend = tmp.begin();
	while (imgend != tmp.end())
	{
		std::advance(imgend, n_rows * n_cols);
		X_train.push_back(UImg(std::vector<_PixelType>(imgbegin, imgend), { 1, n_rows, n_cols }));
		imgbegin = imgend;
	}
	assert(X_train.size() == n);
	return true;
}

bool MNIST::_read_y_test(const std::string &basedir)
{
	if (basedir.back() == '/')
	{
		throw std::invalid_argument("basedir");
	}
	std::string filename(basedir);
	filename.append("/t10k-labels-idx1-ubyte");
	std::ifstream infile(filename, std::ios::binary);
	U_MNISTLabelFileHeader u_header;
	infile.read(u_header.bytes, sizeof(U_MNISTLabelFileHeader));
	if (infile.gcount() < sizeof(U_MNISTLabelFileHeader))
	{
		std::cerr << "read_mnist: error: failed to read bytes\n";
		return false;
	}
	unsigned int magic = ntohl(u_header.h.magic);
	unsigned int n = ntohl(u_header.h.n);
	if (magic != 2049)
	{
		std::cerr << "read_mnist: error: magic number mismatch\n";
		return false;
	}
	if (n != 10000)
	{
		std::cerr << "read_mnist: error: number of items mismatch\n";
		return false;
	}
	std::istreambuf_iterator<char> begin(infile), end;
	std::vector<_LabelType> tmp(begin, end);
	if (tmp.size() != n)
	{
		std::cerr << "read_mnist: error: number of labels mismatch\n";
		return false;
	}
	y_test.swap(tmp);
	return true;
}
bool MNIST::_read_X_test(const std::string &basedir)
{
	if (basedir.back() == '/')
	{
		throw std::invalid_argument("basedir");
	}
	std::string filename(basedir);
	filename.append("/t10k-images-idx3-ubyte");
	std::ifstream infile(filename, std::ios::binary);
	U_MNISTImageFileHeader u_header;
	infile.read(u_header.bytes, sizeof(U_MNISTImageFileHeader));
	if (infile.gcount() < sizeof(U_MNISTImageFileHeader))
	{
		std::cerr << "read_mnist: error: failed to read bytes\n";
		return false;
	}
	unsigned int magic = ntohl(u_header.h.magic);
	unsigned int n = ntohl(u_header.h.n);
	unsigned int n_rows = ntohl(u_header.h.n_rows);
	unsigned int n_cols = ntohl(u_header.h.n_cols);
	if (magic != 2051)
	{
		std::cerr << "read_mnist: error: magic number mismatch\n";
		return false;
	}
	if (n != 10000)
	{
		std::cerr << "read_mnist: error: number of items mismatch\n";
		return false;
	}
	if (n_rows != 28)
	{
		std::cerr << "read_mnist: error: number of rows mismatch\n";
		return false;
	}
	if (n_cols != 28)
	{
		std::cerr << "read_mnist: error: number of columns mismatch\n";
		return false;
	}
	std::istreambuf_iterator<char> begin(infile), end;
	std::vector<_PixelType> tmp(begin, end);
	if (tmp.size() != n * n_rows * n_cols)
	{
		std::cerr << "read_mnist: error: number of pixels mismatch\n";
		return false;
	}

	auto imgbegin = tmp.begin();
	auto imgend = tmp.begin();
	while (imgend != tmp.end())
	{
		std::advance(imgend, n_rows * n_cols);
		X_test.push_back(UImg(std::vector<_PixelType>(imgbegin, imgend), { 1, n_rows, n_cols }));
		imgbegin = imgend;
	}
	return true;
}

void MNIST::read_from_file(const std::string &basedir)
{
	if (!_read_y_train(basedir))
	{
		throw std::runtime_error("failed to read y_train");
	}
	if (!_read_X_train(basedir))
	{
		throw std::runtime_error("failed to read X_train");
	}
	assert(X_train.size() == y_train.size());
	if (!_read_y_test(basedir))
	{
		throw std::runtime_error("failed to read y_test");
	}
	if (!_read_X_test(basedir))
	{
		throw std::runtime_error("failed to read X_test");
	}
	assert(X_test.size() == y_test.size());
}

void normalize(const MNIST &mnist, std::vector<MNIST::DImg> &X_train,
		std::vector<MNIST::DImg> &X_test)
{
	std::vector<MNIST::DImg> orig_X_train = uint8_to_double(mnist.X_train);
	std::vector<MNIST::DImg> orig_X_test = uint8_to_double(mnist.X_test);

	std::vector<double> pixels_train;
	for (const auto &e : orig_X_train)
	{
		std::vector<double> tmp = e.data();
		pixels_train.insert(pixels_train.end(), tmp.begin(), tmp.end());
	}
	double mean = 0.0;
	for (auto e : pixels_train)
	{
		mean += e;
	}
	mean /= pixels_train.size();
	double unbiased_std = 0.0;
	for (auto e : pixels_train)
	{
		unbiased_std += (e - mean) * (e - mean);
	}
	unbiased_std = std::sqrt(unbiased_std / (pixels_train.size() - 1));
	assert(unbiased_std > 0.0);

	for (const auto &e : orig_X_train)
	{
		std::vector<double> pixels = e.data();
		for (auto &x : pixels)
		{
			x = (x - mean) / unbiased_std;
		}
		// DImg ndims=3
		X_train.push_back(MNIST::DImg(std::move(pixels),
		{
			e.shape(0), e.shape(1), e.shape(2)
		}));
	}
	for (const auto &e : orig_X_test)
	{
		std::vector<double> pixels = e.data();
		for (auto &x : pixels)
		{
			x = (x - mean) / unbiased_std;
		}
		X_test.push_back(MNIST::DImg(std::move(pixels),
		{
			e.shape(0), e.shape(1), e.shape(2)
		}));
	}
}

Tsr4 make_batch(const std::vector<MNIST::DImg> &inputs)
{
	std::vector<double> pixels;
	for (const auto &e : inputs)
	{
		std::vector<double> tmp = e.data();
		pixels.insert(pixels.end(), tmp.begin(), tmp.end());
	}
	return Tsr4(std::move(pixels),
	{
		inputs.size(), inputs[0].shape(0), inputs[0].shape(1), inputs[0].shape(2)
	});
}

LabelVec make_batch(const std::vector<size_t> &labels)
{
	return LabelVec(labels, { labels.size() });
}

void reset_batches(size_t n_batch, const std::vector<MNIST::DImg> &inputs,
		const std::vector<size_t> &labels, std::vector<Tsr4> &out_inputs,
		std::vector<LabelVec> &out_labels)
{
	assert(n_batch != 0);
	assert(inputs.size() == labels.size());
	const size_t M = inputs.size();
	out_inputs.clear();
	out_labels.clear();

	auto imverybegin = inputs.begin();
	auto imbatchbegin = inputs.begin();
	auto imbatchend = imbatchbegin;
	std::advance(imbatchend, n_batch);
	auto lbbatchbegin = labels.begin();
	auto lbbatchend = lbbatchbegin;
	std::advance(lbbatchend, n_batch);
	while (std::distance(imverybegin, imbatchend) <= M)
	{
		std::vector<MNIST::DImg> imbatch(imbatchbegin, imbatchend);
		out_inputs.push_back(make_batch(imbatch));
		std::vector<size_t> lbbatch(lbbatchbegin, lbbatchend);
		out_labels.push_back(make_batch(lbbatch));
		imbatchbegin = imbatchend;
		std::advance(imbatchend, n_batch);
		lbbatchbegin = lbbatchend;
		std::advance(lbbatchend, n_batch);
	}
}
