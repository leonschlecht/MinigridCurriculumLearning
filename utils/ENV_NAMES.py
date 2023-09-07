class ENV_NAMES:
    """
    This class contains all the full environment names that are used for the training process
    """
    DOORKEY_5x5 = "MiniGrid-DoorKey-5x5"
    DOORKEY_6x6 = "MiniGrid-DoorKey-6x6"
    DOORKEY_8x8 = "MiniGrid-DoorKey-8x8"
    DOORKEY_10x10 = "MiniGrid-DoorKey-10x10"
    DOORKEY_12x12 = "MiniGrid-DoorKey-12x12"
    DOORKEY_16x16 = "MiniGrid-DoorKey-16x16"

    DYNAMIC_5x5 = "MiniGrid-Dynamic-Obstacles-5x5"
    DYNAMIC_6x6 = "MiniGrid-Dynamic-Obstacles-6x6"
    DYNAMIC_8x8 = "MiniGrid-Dynamic-Obstacles-8x8"
    DYNAMIC_16x16 = "MiniGrid-Dynamic-Obstacles-16x16"

    CUSTOM_POSTFIX = "-custom-diff"

    # TODO find better way so you dont have to manually change everything by hand when sth changes
    # maybe with an additional parameter or something, to determine the correctl ength and which envs etc
    DOORKEY_ENVS = [DOORKEY_6x6,
                    DOORKEY_8x8,
                    DOORKEY_10x10,
                    DOORKEY_12x12
                    ]

    DYNAMIC_OBST_ENVS = [DYNAMIC_5x5,
                         DYNAMIC_6x6,
                         DYNAMIC_8x8,
                         DYNAMIC_16x16
                         ]
