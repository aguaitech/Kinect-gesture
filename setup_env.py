import os

os.environ['OPENNI2_INCLUDE'] = os.path.join(
    os.path.dirname(__file__), 'OpenNI2', 'Include')
os.environ['OPENNI2_REDIST'] = os.path.join(
    os.path.dirname(__file__), 'OpenNI2', 'Redist')
os.environ['OPENNI2_REDIST64'] = os.path.join(
    os.path.dirname(__file__), 'OpenNI2', 'Redist')
os.environ['NITE2_INCLUDE'] = os.path.join(
    os.path.dirname(__file__), 'NiTE', 'Include')
os.environ['NITE2_REDIST'] = os.path.join(
    os.path.dirname(__file__), 'NiTE', 'Redist')
os.environ['NITE2_REDIST64'] = os.path.join(
    os.path.dirname(__file__), 'NiTE', 'Redist')
