PDF=current_static_int8_awq_slides.pdf
TEX=current_static_int8_awq_slides.tex

.PHONY: all clean

all: $(PDF)

$(PDF): $(TEX)
	pdflatex -interaction=nonstopmode -halt-on-error $(TEX)
	pdflatex -interaction=nonstopmode -halt-on-error $(TEX)

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb $(PDF)
