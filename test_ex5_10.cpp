#include "ex5_10.h"

void test_conv2d_valid()
{
	std::vector<double> _arr =
	{
		2, 7, 1, 9, 2, 8, 6, 5, 9, 6, 9, 9, 2, 6, 7, 0, 2, 1, 5, 0, 9, 4,
		5, 4, 3, 9, 9, 7, 5, 3, 3, 6, 6, 9, 2, 4, 0, 0, 3, 6, 2, 0, 7, 3,
		6, 0, 9, 7, 9, 4, 1, 6, 3, 3, 4, 7, 8, 0, 1, 2, 1, 6, 6, 3, 2, 7,
		7, 4, 2, 8, 9, 4
	};
	std::vector<double> _arr2 =
	{
		3, 6, 5, 0, 7, 6, 8, 2, 9, 4, 4, 2, 0, 0, 3, 3, 7, 4, 8, 8, 1, 5,
		6, 7
	};
	std::vector<double> _arr3 = { 2, 9 };
	Tsr4 Z(_arr, { 2, 3, 3, 4 });
	Tsr4 W(_arr2, { 2, 3, 2, 2 });
	Vec b(_arr3, { 2 });
	std::vector<double> _arr4 =
	{
		217, 306, 311, 300, 250, 245, 200, 262, 235, 295, 239, 243,
		184, 206, 273, 211, 232, 291, 234, 245, 249, 229, 234, 246
	};
	Tsr4 A = conv2d_valid(Z, W, b);
	Tsr4 Aref(_arr4, { 2, 2, 2, 3 });
	assert(A.shape(0) == 2);
	assert(A.shape(1) == 2);
	assert(A.shape(2) == 2);
	assert(A.shape(3) == 3);
	for (size_t n = 0; n < 2; ++n)
	{
		for (size_t k = 0; k < 2; ++k)
		{
			for (size_t i = 0; i < 2; ++i)
			{
				for (size_t j = 0; j < 3; ++j)
				{
					assert(A({ n, k, i, j }) == Aref({ n, k, i, j }));
				}
			}
		}
	}
}
