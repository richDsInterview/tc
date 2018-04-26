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
import shutil
import numpy as np
import imageio
import compareimages.core as c

def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('infile', help="path to file to compare") #, type=argparse.FileType('r'))
    parser.add_argument('dbDir', help="path to 'Database' directory")
    parser.add_argument('method', help='', choices=["exact", "scaled", "phash"])
    parser.add_argument('-t', help="make thumbnail image", action="store_true")
    parser.add_argument('-g', help="make greyscale/flatten colour information", action="store_true")
    parser.add_argument('-q', help="pixel quantisation factor 2^x (default 2^8)", type=int, default=8)
    parser.add_argument('--threshold', help="threshold val (default 0.7)", type=float, default=0.7)


    args = parser.parse_args(arguments)
    baseDir = os.path.abspath(args.dbDir)

    match = False
    score = 0.0

    try:
        imTest = imageio.imread(args.infile)
    except FileNotFoundError: # file does not exist
        print("tc: test file does not exist")
        exit()
    except ValueError: # file is not an image file
        print("tc: test file is not an image file")
        exit()
    except Exception: # unexpected/unknown exception
        print("tc: unexpected/unhandled behaviour in image file reading process")
        exit()

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
        if args.method == 'exact':
            score = c.compare_images_exact(args.infile, baseDir + "/" + dbImg)
            #diff = np.array([])
        elif args.method == 'scaled':
            score = c.compare_images_scaled(args.infile, baseDir + "/" + dbImg, thumb=args.t, grey=args.g, pixelQuant=args.q)
        elif args.method  == 'phash':
            score = c.compare_images_phash(args.infile, baseDir + "/" + dbImg)
            #diff = np.array([])

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

