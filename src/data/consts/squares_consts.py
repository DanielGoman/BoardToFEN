# Relevant squares to parse in each type of input board
BOARD_SIDE_SIZE = 8
RELEVANT_SQUARES = {'full_boards': {'rows': range(BOARD_SIDE_SIZE),
                                    'cols': range(BOARD_SIDE_SIZE)
                                    },
                    'replaced_king_queen': {'rows': [0, 7],
                                            'cols': [BOARD_SIDE_SIZE//2 - 1, BOARD_SIDE_SIZE//2]
                                            }
                    }
