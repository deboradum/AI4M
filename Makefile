red:=$(shell tput bold ; tput setaf 1)
green:=$(shell tput bold ; tput setaf 2)
yellow:=$(shell tput bold ; tput setaf 3)
blue:=$(shell tput bold ; tput setaf 4)
magenta:=$(shell tput bold ; tput setaf 5)
cyan:=$(shell tput bold ; tput setaf 6)
reset:=$(shell tput sgr0)


data/TOY:
	python gen_toy.py --dest $@ -n 10 10 -wh 256 256 -r 50

data/TOY2:
	rm -rf $@_tmp $@
	python scripts/gen_two_circles.py --dest $@_tmp -n 1000 100 -r 25 -wh 256 256
	mv $@_tmp $@


# Extraction and slicing for Segthor
## Original one
data/segthor_part1: data/segthor_part1.zip
	$(info $(yellow)unzip $<$(reset))
	sha256sum -c data/segthor_part1.sha256
	unzip -q $<
	rm -f $@/.DS_STORE

data/SEGTHOR:
	$(info $(green)python $(CFLAGS) slice_segthor.py$(reset))
	rm -rf $@_tmp $@
	python $(CFLAGS) slice_segthor.py --source_dir data/segthor_part1 --dest_dir $@_tmp \
		--shape 256 256 --retain 5
	mv $@_tmp $@

data/SEGTHOR_RESAMPLED:
	$(info $(green)python $(CFLAGS) slice_segthor.py (Resampled)$(reset))
	rm -rf $@_tmp $@
	python $(CFLAGS) slice_segthor.py --source_dir data/segthor_part1 --dest_dir $@_tmp \
		--shape 256 256 --retains 5 --resample --target_spacing 1.0 1.0 2.5
	mv $@_tmp $@


# Slicing for the aorta-recovered SegTHOR
## data/SEGTHOR_aorta holds the challenge's four organs: class 1 of the archive's
## GT is esophagus UNION aorta (folded together), so retrieve_aorta.py splits it
## back with a distance-transform watershed and gates. See
## archive/docs/aorta-findings.md. The slicing uses the E001-E016 split
## (retains 5, seed 0, fold 0) so new runs stay comparable with the old ones.
## Note the esophagus target changes: it is the thin organ alone now, not the fold.
AORTA_PY ?= ./ai4mi/bin/python
AORTA_SLICE = $(AORTA_PY) slice_segthor.py --source_dir data/SEGTHOR_aorta --dest_dir $@_tmp \
	--shape 256 256 --retains 5 --seed 0 --fold 0 -p -1

data/SEGTHOR_aorta: data/segthor_part1
	$(info $(green)python archive/aorta_recovery/retrieve_aorta.py$(reset))
	$(AORTA_PY) archive/aorta_recovery/retrieve_aorta.py --src data/segthor_part1/train --dest $@ \
		--gt2 data/segthor_part1/train/Patient_07/GT2.nii.gz --figures analysis/figures/aorta_inspection

data/SEGTHOR_aorta_norm: data/SEGTHOR_aorta
	$(info $(green)python slice_segthor.py (per-patient min-max)$(reset))
	rm -rf $@_tmp $@
	$(AORTA_SLICE)
	mv $@_tmp $@

data/SEGTHOR_aorta_huwide: data/SEGTHOR_aorta
	$(info $(green)python slice_segthor.py (window -1000 1000)$(reset))
	rm -rf $@_tmp $@
	$(AORTA_SLICE) --window -1000 1000
	mv $@_tmp $@

data/SEGTHOR_aorta_husoft: data/SEGTHOR_aorta
	$(info $(green)python slice_segthor.py (window -200 300)$(reset))
	rm -rf $@_tmp $@
	$(AORTA_SLICE) --window -200 300
	mv $@_tmp $@

data/SEGTHOR_aorta_resampled: data/SEGTHOR_aorta
	$(info $(green)python slice_segthor.py (window -1000 1000, resampled 1.0 1.0 2.5)$(reset))
	rm -rf $@_tmp $@
	$(AORTA_SLICE) --window -1000 1000 --resample --target_spacing 1.0 1.0 2.5
	mv $@_tmp $@

# Full SegTHOR training set: 40 patients, all four organs labelled by the course
# (released 2026-09-24, readme link). Supersedes segthor_part1 and the aorta recovery.
# Split: the stratified 32/8 split in configs/splits/ (scripts/make_split.py). Window -1000..1000 HU (E014 result).
data/segthor_train_full: data/segthor_train_full.zip
	$(info $(yellow)unzip $<$(reset))
	rm -rf $@_tmp $@ && mkdir -p $@_tmp
	unzip -q $< -d $@_tmp
	mv $@_tmp $@

data/SEGTHOR_FULL_huwide: data/segthor_train_full
	$(info $(green)python slice_segthor.py (full set, window -1000 1000)$(reset))
	rm -rf $@_tmp $@
	python $(CFLAGS) slice_segthor.py --source_dir data/segthor_train_full --dest_dir $@_tmp \
		--shape 256 256 --split_file configs/splits/segthor_full_32_8.json -p -1 --window -1000 1000
	mv $@_tmp $@
