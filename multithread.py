# -*- coding: utf-8 -*-
"""
Multithread image processing
Created on Wed Oct 27 17:56:19 2021



@author: 56153805
"""

def get_drr_balls_pool(name, i, num_of_balls):
    """
    Get the ball positions from the drr image.

    Parameters
    ----------
    files : list of strings
        contains the names of tif image files.

    Returns
    -------
    None.

    """
    
    filename = drrfolder / name
    
    
    im = imageio.imread(filename)
    im = np.array(im)
    im = sparse_image(im, cropx, cropy)
       
    thresh = threshold_otsu(im)
    binary = im > thresh
    
    edges = canny(im, sigma=3, low_threshold=10, high_threshold=11, mask=binary)
    
    
    hough_radii = np.arange(4,20,3)
    hough_res = hough_circle(edges, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                        total_num_peaks=num_of_balls)
    
    ball_positions = list(zip(cx,cy))

    # order balls by y position.
    
    ball_positions = sorted(ball_positions, key=itemgetter(1)) 
    
    # p
    ball_positions = [item for t in ball_positions for item in t]
        

    # try:
    #     df.at[i, 'DRRBalls'] = ball_positions
    #         # df.at[] is faster than df.loc[] but will preserve data 
    #         # type of df series. And it will do it silently. Saving 
    #         # floats in int columns will be lossful.
    # except AssertionError as error:
    #     print(error)
    #     print("Probably found too many balls or apertures." + 
    #           "Change detection settings")
        
    # #debug lines
    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    # img = color.gray2rgb(im)
    # for center_y, center_x, radius in zip(cy, cx, radii):
    #     circy, circx = circle_perimeter(center_y, center_x, radius,
    #                             shape=im.shape)
    #     img[circy, circx] = (120, 20, 1)
    #     ax.plot(center_x, center_y, color="darkred", marker='o', 
    #                     linewidth=3, markersize=3)
    # ax.imshow(img, cmap=plt.cm.RdBu)
    # plt.show()
    # print(hough_radii)
    # print('x1 y1 {} \nx2 y2 {} \nx3 y3 {} \nx4, y4 {}'.format(ball_positions[0:2],ball_positions[2:4],ball_positions[4:6],ball_positions[6:8]))
    # #end debug lines
    
    
    # Progress bar
    # text = "Finding balls in DRR {0}".format(n)
   
    # update_progress(i/progmax, text)
    
def main():
    """
    Suppress execution on import.
    
    Returns
    -------
    None.

    """
    pass

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()