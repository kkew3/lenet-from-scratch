#!/bin/bash
set -e

requires() {
	if ! command -v "$1" > /dev/null; then
		echo "require \`$1' but found none" >&2
		exit 1
	fi
}

requires curl
requires gunzip


files=(
	train-images-idx3-ubyte
	train-labels-idx1-ubyte
	t10k-images-idx3-ubyte
	t10k-labels-idx1-ubyte
)
baseurl="http://yann.lecun.com/exdb/mnist/"
for f in ${files[@]}; do
	if [ ! -f "$f" ]; then
		echo "Downloading to $f ..."
		curl -s "${baseurl}${f}.gz" | gunzip > "$f"
	fi
done
