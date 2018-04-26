#!/usr/bin/env python

""" Richard Paul Armstrong 2018
A simple python script to exactly compare an image to all the images contained in a database
directory. The comparison should be sensitive to scaling and assumes, as per the question,
that the scaling will be performed with common ratio.

Arguments:
    infile : path to image file to compare
    dbDir : path to database directory containing comparison images
"""
import sys
import argparse
import os
import numpy as np
import shutil
import compareimages.core as c

def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('infile', help="path to file to compare") #, type=argparse.FileType('r'))
    parser.add_argument('dbDir', help="path to 'Database' directory")
    parser.add_argument('-m', '--method', help='', type=string, choices=['exact', 'scaled', 'phash'], default='phash')
    parser.add_argument('-t', help="make thumbnail image", action="store_true")
    parser.add_argument('-g', help="make greyscale/flatten colour information", action="store_true")
    parser.add_argument('-q', help="pixel quantisation factor 2^x (default 2^8)", type=int, default=8)
    parser.add_argument('--threshold', help="threshold val (default 0.7)", type=float, default=0.7)


    args = parser.parse_args(arguments)
    baseDir = os.path.abspath(args.dbDir)

    match = False
    score = 0.0

    try:
        imTest = imageio.imread(infile)
    except FileNotFoundError: # file does not exist
        print("compare_images_phash: one of the file arguments does not exist")
        return None
    except ValueError: # file is not an image file
        print("compare_images_exact: one of the file arguments is not an image file")
        return None
    except Exception: # unexpected/unknown exception
        print("compare_images_exact: unexpected/unhandled behaviour in image file reading process")
        return None

    # loop through each image in the Test database
    for dbImg in os.listdir(args.dbDir):
        # convert current DB image into numpy array
        try:
            imDB = imageio.imread(baseDir + "/" + dbImg)
        except ValueError:  # file in database is not an image file: just ignore it
            continue
        except Exception:  # unexpected/unknown exception: continue
            continue

        # compare the two images
        if args.m == 'exact':
            score = c.compare_images_exact(imDB, imTest)
            diff = np.array([])
        elif args.m == 'scaled':
            score, diff = c.compare_images_scaled(imDB, imTest, thumb=args.t, grey=args.g, pixelQuant=args.q)
        elif args.m == 'phash':
            score, diff = c.compare_images_phash(imDB, imTest)
            diff = np.array([])

        if score >= args.threshold:
            match = True
            # print the result
            print("Matching image found in database directory: ", dbImg, " (score: ", str(score), ")")


    if match:
        print("Match(es) found. Not adding")
    else:
        print("No match found. Adding ", args.infile, " to database directory: ", baseDir)
        shutil.copy(args.infile, baseDir)



if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

