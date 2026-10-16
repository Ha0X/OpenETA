"""Keep private USD/filesystem identities out of operation rejection feedback."""
import hashlib
import re


def public_error_message(exc):
    text=str(exc)
    pattern=r"/(?:World|home|tmp|media|mnt|opt|usr|var)(?:/[^\s'\"\],)};]+)+"
    def opaque(match):
        return 'private_geometry_'+hashlib.sha256(match.group().encode()).hexdigest()[:10]
    return re.sub(pattern,opaque,text)
