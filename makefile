run:
	python -W ignore experiment0.py
	cd article && pdflatex main.tex main.pdf

pdf:
	cd article && pdflatex main.tex main.pdf

analysis:
	python -W ignore analysis0.py
	cd article && pdflatex main.tex main.pdf
