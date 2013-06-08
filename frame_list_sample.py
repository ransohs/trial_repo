''' local list of tail numbers with mapping to LFL.  FDS also has a web API'''

def build_frame_list(logger):
    frame_list={}

    frame_list['N001WB'] = {'Frame': 'Wright_Flyer_1',
                            'Manufacturer': 'Wright Brothers',
                            'Precise Positioning': False,
                            'Series': 'Flyer_1',
                            'Family': 'Flyer',
                            'Frame Doubled' : True}    
    
    # Copy the registration into the Tail Number field to avoid double entry.
    for each_frame in frame_list:
        frame_list[each_frame]['Tail Number'] = each_frame
    
    logger.debug('Have local frame_list')
    return frame_list