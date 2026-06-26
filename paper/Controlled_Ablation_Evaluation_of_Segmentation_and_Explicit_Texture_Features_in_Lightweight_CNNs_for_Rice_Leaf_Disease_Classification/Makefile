PDF=build/main.pdf

all:
	latexmk -pdf -interaction=nonstopmode -outdir=build main.tex

clean:
	latexmk -C -outdir=build main.tex
	rm -rf build/*
