class ENV_NAMES:
    DOORKEY_4x4 = "MiniGrid-DoorKey-4x4"
    DOORKEY_5x5 = "MiniGrid-DoorKey-5x5"
    DOORKEY_6x6 = "MiniGrid-DoorKey-6x6"
    DOORKEY_7x7 = "MiniGrid-DoorKey-7x7"
    DOORKEY_8x8 = "MiniGrid-DoorKey-8x8"
    DOORKEY_9x9 = "MiniGrid-DoorKey-9x9"
    DOORKEY_10x10 = "MiniGrid-DoorKey-10x10"
    DOORKEY_12x12 = "MiniGrid-DoorKey-12x12"
    DOORKEY_16x16 = "MiniGrid-DoorKey-16x16"

    CUSTOM_POSTFIX = "-custom-diff"

    # TODO find better way so you dont have to manually change everything by hand when sth changes
    # maybe with an additional parameter or something, to determine the correctl ength and which envs etc
    ALL_ENVS = [DOORKEY_6x6,
                DOORKEY_8x8,
                DOORKEY_10x10,
                DOORKEY_12x12
                ]

    DOORKEY_5x5_v0 = DOORKEY_5x5 + "-v0"
