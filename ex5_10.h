#ifndef _EX5_10_H_
#define _EX5_10_H_

#include <algorithm>
#include <arpa/inet.h>
#include <array>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// why not use template here? because it will otherwise not be able to inline

inline size_t indexcont1(std::array<size_t, 1> I, std::array<size_t, 1> N)
{
	return I[0];
}

inline size_t indexcont2(std::array<size_t, 2> I, std::array<size_t, 2> N)
{
	return I[1] + I[0] * N[1];
}

inline size_t indexcont3(std::array<size_t, 3> I, std::array<size_t, 3> N)
{
	return I[2] + (I[1] + I[0] * N[1]) * N[2];
}

inline size_t indexcont4(std::array<size_t, 4> I, std::array<size_t, 4> N)
{
	return I[3] + (I[2] + (I[1] + I[0] * N[1]) * N[2]) * N[3];
}

/// TensorNd
template <typename ValueType, size_t dim>
struct Tsr
{
	Tsr() = delete;
	Tsr(const std::array<size_t, dim> &shape)
		: _shape { shape }
	{
		size_t size = 1;
		for (size_t i = 0; i < dim; ++i)
		{
			size *= shape[i];
		}
		_arr.resize(size);
	}
	Tsr(const Tsr &other)
		: _arr(other._arr)
		, _shape { other._shape }
	{
	}
	Tsr(Tsr &&other)
	{
		_arr.swap(other._arr);
		_shape.swap(other._shape);
	}
	explicit Tsr(const std::vector<ValueType> &arr,
			const std::array<size_t, dim> &shape)
		: _arr(arr)
		, _shape { shape }
	{
#ifndef NDEBUG
		size_t n = 1;
		for (size_t i = 0; i < dim; ++i)
		{
			n *= shape[i];
		}
		assert(arr.size() == n);
#endif
	}
	explicit Tsr(std::vector<ValueType> &&arr,
			const std::array<size_t, dim> &shape)
		: _arr(arr)
		, _shape { shape }
	{
#ifndef NDEBUG
		size_t n = 1;
		for (size_t i = 0; i < dim; ++i)
		{
			n *= shape[i];
		}
		assert(arr.size() == n);
#endif
	}
	~Tsr() = default;
	Tsr &operator=(const Tsr &other)
	{
		std::vector<ValueType> tmp(other._arr);
		_arr.swap(tmp);
		for (size_t i = 0; i < dim; ++i)
		{
			_shape[i] = other._shape[i];
		}
		return *this;
	}
	// Tested
	inline ValueType &operator()(const std::array<size_t, dim> &coor)
	{
		size_t i = coor[0];
		for (size_t c = 1; c < dim; ++c)
		{
			i = coor[c] + i * _shape[c];
		}
#ifndef NDEBUG
		return _arr.at(i);
#else
		return _arr[i];
#endif
	}
	// Tested
	inline const ValueType &operator()(const std::array<size_t, dim> &coor) const
	{
		size_t i = coor[0];
		for (size_t c = 1; c < dim; ++c)
		{
			i = coor[c] + i * _shape[c];
		}
#ifndef NDEBUG
		return _arr.at(i);
#else
		return _arr[i];
#endif
	}
	// Tested
	inline size_t shape(size_t i) const
	{
		return _shape[i];
	}
	inline size_t size() const
	{
		size_t s = 1;
		for (size_t i = 0; i < dim; ++i)
		{
			s *= _shape[i];
		}
		return s;
	}
	// Tested
	std::vector<ValueType> data() const
	{
		std::vector<ValueType> copy(_arr);
		return copy;
	}
	// Tested
	Tsr clone() const
	{
		return Tsr(std::move(data()), { _shape });
	}

	void zero_()
	{
		for (auto &e : _arr)
		{
			e = static_cast<ValueType>(0);
		}
	}

	template <typename Generator>
	void randn_(Generator &gen, double mean = 0.0, double stddev = 1.0)
	{
		std::normal_distribution<> d(mean, stddev);
		for (auto &e : _arr)
		{
			e = static_cast<ValueType>(d(gen));
		}
	}

private:
	std::vector<ValueType> _arr;
	std::array<size_t, dim> _shape;
};

using Vec = Tsr<double, 1>;
using Mat = Tsr<double, 2>;
using Tsr4 = Tsr<double, 4>;
using LabelVec = Tsr<size_t, 1>;

/// Equivalent to numpy.dot on 1d array
double dot(const Vec &, const Vec &);
/// Equivalent to numpy.dot on 2d array
Mat matmul(const Mat &, const Mat &);
/// (N, C) + (N,), broadcast on axis=1
void mvadd_(Mat &, const Vec &);
/// (N, C) + (C,), broadcast on axis=0
void mvadd2_(Mat &, const Vec &);
/// Equivalent to numpy.tile(mat, (1, n))
Mat htile(const Mat &, size_t);
/// 2d VALID convolution
/// @param X the inputs, of shape (N, C, H, W)
/// @param u the weights, of shape (D, C, K, K)
/// @param b the biases, of shape (D,)
/// @return of shape (N, D, H-K+1, W-K+1)
Tsr4 conv2d_valid(const Tsr4 &, const Tsr4 &, const Vec &);
/// @param X the inputs, of shape (N, C, H, W)
/// @param p the padding size
/// @return of shape (N, C, H+2p, W+2p)
Tsr4 pad2d(const Tsr4 &, size_t);
/// return the result and the mask
std::pair<Tsr4, Tsr4> maxpool2d(const Tsr4 &, size_t);
/// flatten Tsr4 to Mat
Mat mflatten(const Tsr4 &);
/// ReLU inplace
void relu_(Tsr4 &);
/// ReLU inplace
void relu_(Mat &);
/// Fully connected layer.
/// @param X the inputs, of shape (N, C)
/// @param w the weights, of shape (C, D)
/// @param b the biases, of shape (D,)
/// @return of shape (N, D)
Mat linear(const Mat &, const Mat &, const Vec &);
/// Apply softmax over each row of input matrix
Mat softmax(const Mat &);
/// Cross entropy loss, reduction=mean
/// @param z activations after softmax but before log, of shape (N, C)
/// @param t labels, of shape (N,), label values should be less than C
double celoss(const Mat &, const LabelVec &);
/// Cross entropy loss backward, reduction=mean
/// @param z activations after softmax but before log, of shape (N, C)
/// @param t labels, of shape (N,), label values should be less than C
/// @return the derivative of loss with respect to `z`
Mat b_celoss(const Mat &, const LabelVec &);
/// Softmax backward.
/// @param da the derivative of loss with respect to the softmax outputs
/// @param a the softmax outputs
/// @param t labels, of shape (N,), label values should be less than C
/// @return the derivative of loss with respect to `z`
Mat b_softmax(const Mat &, const Mat &, const LabelVec &);
/// Fully connected layer backward.
/// @param da the derivative of loss with respect to the layer outputs
/// @param X the inputs
/// @param w the weights
/// @return tuple of the derivative of loss with respect to `X`, the derivative
///         of loss with respect to `w`, the derivative of loss with respect to
///         the bias
std::tuple<Mat, Mat, Vec> b_linear(const Mat &, const Mat &, const Mat &);
/// ReLU backward.
/// @param da the derivative of loss with respect to the ReLU outputs
/// @param a the ReLU outputs
/// @return the derivative of loss with respect to the ReLU inputs
Mat b_relu(const Mat &, const Mat &);
Tsr4 b_relu(const Tsr4 &, const Tsr4 &);
/// Reconstruct Tsr4 from Mat.
Tsr4 b_mflatten(const Mat &, const std::array<size_t, 4> &);
/// pad2d backward.
/// @param da the derivative of loss with respect to the pad2d outputs
/// @param p padding size
/// @return the derivative of loss with respect to the pad2d inputs
Tsr4 b_pad2d(const Tsr4 &, size_t);
/// maxpool2d backward.
/// @param da the derivative of loss with respect to the maxpool2d outputs
/// @param m the maxpool2d output masks
/// @param p the pooling size
/// @return the derivative of loss with respect to the maxpool2d inputs
Tsr4 b_maxpool2d(const Tsr4 &, const Tsr4 &, size_t);
/// conv2d_valid backward.
/// @param da the derivative of loss with respect to the conv2d_valid outputs
/// @param z the conv2d_valid inputs
/// @param w the conv2d_valid filter weights
/// @return tuple of the derivative of loss with respect to `z`, the derivative
///         of loss with respect to `w`, the derivative of loss with respect to
///         the bias
std::tuple<Tsr4, Tsr4, Vec> b_conv2d_valid(const Tsr4 &, const Tsr4 &,
		const Tsr4 &);
/// Equivalent to m1 -= scale * m2
/// @param m1 tensor 1, to be subtracted from
/// @param m2 tensor 2
/// @param scale
void tsrsub_(Tsr4 &, const Tsr4 &, double = 1.0);
void tsrsub_(Mat &, const Mat &, double = 1.0);
void tsrsub_(Vec &, const Vec &, double = 1.0);
/// Equivalent to numpy.argmax(..., axis=1)
LabelVec margmax(const Mat &);
/// Compare number of matches between two LabelVec
size_t n_matches(const LabelVec &, const LabelVec &);

struct LeNetParams
{
	Tsr4 conv1_w;  // 6 x 3 x 5 x 5
	Vec conv1_b;   // 6
	Tsr4 conv2_w;  // 16 x 6 x 5 x 5
	Vec conv2_b;   // 16
	Mat linear1_w; // 1024 x 120
	Vec linear1_b; // 120
	Mat linear2_w; // 120 x 84
	Vec linear2_b; // 84
	Mat linear3_w; // 84 x 10
	Vec linear3_b; // 10

	Tsr4 grad_conv1_w;
	Vec grad_conv1_b;
	Tsr4 grad_conv2_w;
	Vec grad_conv2_b;
	Mat grad_linear1_w;
	Vec grad_linear1_b;
	Mat grad_linear2_w;
	Vec grad_linear2_b;
	Mat grad_linear3_w;
	Vec grad_linear3_b;

	LeNetParams()
		: conv1_w({ 6, 1, 5, 5 })
	, conv1_b({ 6 })
	, conv2_w({ 16, 6, 5, 5 })
	, conv2_b({ 16 })
	, linear1_w({ 784, 120 })
	, linear1_b({ 120 })
	, linear2_w({ 120, 84 })
	, linear2_b({ 84 })
	, linear3_w({ 84, 10 })
	, linear3_b({ 10 })
	, grad_conv1_w({ 6, 3, 5, 5 })
	, grad_conv1_b({ 6 })
	, grad_conv2_w({ 16, 6, 5, 5 })
	, grad_conv2_b({ 16 })
	, grad_linear1_w({ 784, 120 })
	, grad_linear1_b({ 120 })
	, grad_linear2_w({ 120, 84 })
	, grad_linear2_b({ 84 })
	, grad_linear3_w({ 84, 10 })
	, grad_linear3_b({ 10 })
	{
	}

	template <typename Generator>
	void init(Generator &g)
	{
		// stddev=0.1 tested to make activations with inputs(mean=0, stddev=1)
		// stable
		conv1_w.randn_(g, 0.0, 0.1);
		conv1_b.zero_();
		conv2_w.randn_(g, 0.0, 0.1);
		conv2_b.zero_();
		linear1_w.randn_(g, 0.0, 0.1);
		linear1_b.zero_();
		linear2_w.randn_(g, 0.0, 0.1);
		linear2_b.zero_();
		linear3_w.randn_(g, 0.0, 0.1);
		linear3_b.zero_();
		grad_conv1_w.zero_();
		grad_conv1_b.zero_();
		grad_conv2_w.zero_();
		grad_conv2_b.zero_();
		grad_linear1_w.zero_();
		grad_linear1_b.zero_();
		grad_linear2_w.zero_();
		grad_linear2_b.zero_();
		grad_linear3_w.zero_();
		grad_linear3_b.zero_();
	}

	void update(double lr)
	{
		tsrsub_(conv1_w, grad_conv1_w, lr);
		tsrsub_(conv1_b, grad_conv1_b, lr);
		tsrsub_(conv2_w, grad_conv2_w, lr);
		tsrsub_(conv2_b, grad_conv2_b, lr);
		tsrsub_(linear1_w, grad_linear1_w, lr);
		tsrsub_(linear1_b, grad_linear1_b, lr);
		tsrsub_(linear2_w, grad_linear2_w, lr);
		tsrsub_(linear2_b, grad_linear2_b, lr);
		tsrsub_(linear3_w, grad_linear3_w, lr);
		tsrsub_(linear3_b, grad_linear3_b, lr);
	}
};

/// The LeNet, including softmax
/// @param z the inputs
/// @param params the weights and biases
/// @param t the labels
/// @param to_grad whether to back propagate errors and update parameters
/// @return final output of the network (softmax output) and the ce-loss
std::pair<Mat, double> lenet(const Tsr4 &, LeNetParams &, const LabelVec &,
		bool);

struct MNISTLabelFileHeader
{
	unsigned int magic;
	unsigned int n;
};

union U_MNISTLabelFileHeader
{
	MNISTLabelFileHeader h;
	char bytes[sizeof(MNISTLabelFileHeader)];
};

struct MNISTImageFileHeader
{
	unsigned int magic;
	unsigned int n;
	unsigned int n_rows;
	unsigned int n_cols;
};

union U_MNISTImageFileHeader
{
	MNISTImageFileHeader h;
	char bytes[sizeof(MNISTImageFileHeader)];
};

struct MNIST
{
private:
	using _PixelType = unsigned char;
	using _LabelType = size_t;

public:
	using UImg = Tsr<_PixelType, 3>;
	using DImg = Tsr<double, 3>;

	std::vector<UImg> X_train;
	std::vector<UImg> X_test;
	std::vector<_LabelType> y_train;
	std::vector<_LabelType> y_test;

	void read_from_file(const std::string &);

private:
	// Tested
	bool _read_y_train(const std::string &);
	// Tested
	bool _read_X_train(const std::string &);
	bool _read_y_test(const std::string &);
	bool _read_X_test(const std::string &);
};

template <typename ValueType, size_t dim>
std::vector<Tsr<double, dim>> uint8_to_double(const
				std::vector<Tsr<ValueType, dim>> &u)
{
	std::vector<Tsr<double, dim>> v;
	for (const auto &t : u)
	{
		std::array<size_t, dim> shape;
		for (size_t i = 0; i < dim; ++i)
		{
			shape[i] = t.shape(i);
		}
		std::vector<ValueType> tmp = t.data();
		v.push_back(Tsr<double, dim>(std::vector<double>(tmp.begin(), tmp.end()),
						shape));
	}
	return v;
}

/// Normalize pixels to mean=0, stddev=1. The mean and stddev
/// are computed from training set and applied on test set.
/// @param mnist the original dataset
/// @param X_train the normalized mnist.X_train
/// @param X_test the normalized mnist.X_test
void normalize(const MNIST &, std::vector<MNIST::DImg> &,
		std::vector<MNIST::DImg> &);

struct NormalizedMNIST
{
	std::vector<MNIST::DImg> X_train;
	std::vector<size_t> y_train;
	std::vector<MNIST::DImg> X_test;
	std::vector<size_t> y_test;

	NormalizedMNIST() = delete;

	/// Move vectors to this struct by swapping
	NormalizedMNIST(std::vector<MNIST::DImg> &X_train, std::vector<size_t> y_train,
			std::vector<MNIST::DImg> &X_test, std::vector<size_t> y_test)
	{
		this->X_train.swap(X_train);
		this->y_train.swap(y_train);
		this->X_test.swap(X_test);
		this->y_test.swap(y_test);
	}

	// Tested
	template <typename URBG>
	void shuffle_trainset(URBG &&g)
	{
		assert(X_train.size() == y_train.size());
		const size_t n = X_train.size();
		assert(n > 0);
		std::array<size_t, 3> shape =
		{
			X_train[0].shape(0), X_train[0].shape(1), X_train[0].shape(2)
		};
		std::vector<size_t> indices(n);
		for (size_t i = 0; i < n; ++i)
		{
			indices[i] = i;
		}
		std::shuffle(indices.begin(), indices.end(), g);
		std::vector<MNIST::DImg> new_X_train;
		std::vector<size_t> new_y_train;
		for (size_t i = 0; i < n; ++i)
		{
			new_X_train.push_back(std::move(X_train[indices[i]]));
			new_y_train.push_back(y_train[indices[i]]);
		}
		X_train.swap(new_X_train);
		y_train.swap(new_y_train);
	}
};

/// Make a mini-batch of inputs.
Tsr4 make_batch(const std::vector<MNIST::DImg> &);
/// Make a mini-batch of labels
LabelVec make_batch(const std::vector<size_t> &);
/// Clear and populate vector of batches. The last few instances less than the
/// batch size will not be included in the newly formed batches.
/// @param n_batch batch size
/// @param inputs image tensors
/// @param labels the labels
/// @param out_inputs output image batches for network input
/// @param out_labels output label batches for network input
void reset_batches(size_t, const std::vector<MNIST::DImg> &,
		const std::vector<size_t> &, std::vector<Tsr4> &, std::vector<LabelVec> &);

struct TrainRecord
{
	std::vector<double> batch_train_losses;
	std::vector<double> batch_train_accuracies;
	std::vector<double> epoch_train_accuracies;
	std::vector<double> batch_test_losses;
	std::vector<double> batch_test_accuracies;
	std::vector<double> epoch_test_accuracies;

	void write_to_file(const std::string &filename)
	{
		std::ofstream outfile(filename);
		outfile << "batch_train_losses ";
		std::copy(batch_train_losses.begin(), batch_train_losses.end(),
				std::ostream_iterator<double>(outfile, " "));
		outfile << '\n';
		outfile << "batch_train_accuracies ";
		std::copy(batch_train_accuracies.begin(), batch_train_accuracies.end(),
				std::ostream_iterator<double>(outfile, " "));
		outfile << '\n';
		outfile << "epoch_train_accuracies ";
		std::copy(epoch_train_accuracies.begin(), epoch_train_accuracies.end(),
				std::ostream_iterator<double>(outfile, " "));
		outfile << '\n';
		outfile << "batch_test_losses ";
		std::copy(batch_test_losses.begin(), batch_test_losses.end(),
				std::ostream_iterator<double>(outfile, " "));
		outfile << '\n';
		outfile << "batch_test_accuracies ";
		std::copy(batch_test_accuracies.begin(), batch_test_accuracies.end(),
				std::ostream_iterator<double>(outfile, " "));
		outfile << '\n';
		outfile << "epoch_test_accuracies ";
		std::copy(epoch_test_accuracies.begin(), epoch_test_accuracies.end(),
				std::ostream_iterator<double>(outfile, " "));
		outfile << '\n';
	}
};

/// Train LeNet
/// @param n_epoch number of epochs to train
/// @param base_lr initial learning rate
/// @param n_batch the training batch size
/// @param g random number generator
/// @param params the parameters to optimized
/// @param norm_mnist the normalized dataset
/// @param train_losses trainset losses
/// @param train_acces trainset accuracies
/// @param test_losses testset losses
/// @param test_acces testset accuracies
template <typename URBG>
void train(size_t n_epoch, double base_lr, URBG &&g,
		LeNetParams &params, NormalizedMNIST &norm_mnist, TrainRecord &rep)
{
	std::vector<Tsr4> X_train, X_test;
	std::vector<LabelVec> y_train, y_test;
	double lr;
	size_t tt = 0;
	for (size_t ep = 0; ep < n_epoch; ++ep)
	{
		// train and eval on trainset
		norm_mnist.shuffle_trainset(g);
		reset_batches(64, norm_mnist.X_train, norm_mnist.y_train, X_train,
				y_train);
		assert(X_train.size() == y_train.size());
		// so that pop preserve batches order
		std::reverse(X_train.begin(), X_train.end());
		std::reverse(y_train.begin(), y_train.end());
		size_t n_correct_train = 0, n_total_train = 0, n_correct_train_batch = 0;
		while (!X_train.empty())
		{
			lr = base_lr / std::sqrt(static_cast<double>(tt++ / 50 + 1));

			Tsr4 z(X_train.back());
			LabelVec t(y_train.back());
			X_train.pop_back();
			y_train.pop_back();

			auto out = lenet(z, params, t, true);
			params.update(lr);

			n_total_train += t.shape(0);
			n_correct_train_batch = n_matches(margmax(out.first), t);
			rep.batch_train_accuracies.push_back(static_cast<double>
					(n_correct_train_batch) / t.shape(0));
			n_correct_train += n_correct_train_batch;
			rep.batch_train_losses.push_back(out.second);

			std::cout << "train: batch train loss/acc: "
					<< rep.batch_train_losses.back() << '/'
					<< rep.batch_train_accuracies.back() << std::endl;
		}
		rep.epoch_train_accuracies.push_back(static_cast<double>
				(n_correct_train) / n_total_train);
		std::cout << "train: epoch train acc: "
				<< rep.epoch_train_accuracies.back() << std::endl;

		// eval on testset
		reset_batches(1, norm_mnist.X_test, norm_mnist.y_test, X_test, y_test);
		assert(X_test.size() == y_test.size());
		std::reverse(X_test.begin(), X_test.end());
		std::reverse(y_test.begin(), y_test.end());
		size_t n_correct_test = 0, n_total_test = 0, n_correnct_test_batch = 0;
		while (!X_test.empty())
		{
			Tsr4 z(X_test.back());
			LabelVec t(y_test.back());
			X_test.pop_back();
			y_test.pop_back();

			auto out = lenet(z, params, t, false);

			n_total_test += t.shape(0);
			n_correnct_test_batch = n_matches(margmax(out.first), t);
			rep.batch_test_accuracies.push_back(static_cast<double>
					(n_correnct_test_batch) / t.shape(0));
			n_correct_test += n_correnct_test_batch;
			rep.batch_test_losses.push_back(out.second);

			std::cout << "train: batch test loss/acc: "
					<< rep.batch_test_losses.back() << '/'
					<< rep.batch_test_accuracies.back() << std::endl;
		}
		rep.epoch_test_accuracies.push_back(static_cast<double>(n_correct_test) /
				n_total_test);
		std::cout << "train: epoch test acc: "
				<< rep.epoch_test_accuracies.back()
				<< "\n---" << std::endl;
	}
}

#endif // _EX5_10_H_
