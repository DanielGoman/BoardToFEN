from enum import Enum

ACTIVE_COLOR_MAPPING = {False: 'w',
                        True: 'b'}

CASTLING_RIGHTS_MAPPING = {0: 'K',
                           1: 'Q',
                           2: 'k',
                           3: 'q'}


class FenPartDefaults(Enum):
    str = '-'
    int = 0


class Domains(Enum):
    chess = 0
    lichess = 1
    pure_fen = 2


DOMAINS_MAPPING = {
    Domains.chess.value: {
        'domain': 'www.chess.com',
        'prefix': '/analysis?fen=',
        'fen_rows_connector': '%2F',
        'fen_parts_connector': '+',
        'suffix': '&tab=analysis'
    },
    Domains.lichess.value: {
        'domain': 'www.lichess.org',
        'prefix': '/analysis/',
        'fen_rows_connector': '/',
        'fen_parts_connector': '_',
        'suffix': ''
    },
    Domains.pure_fen.value: {
        'fen_rows_connector': '/',
        'fen_parts_connector': '_',
    }
}
