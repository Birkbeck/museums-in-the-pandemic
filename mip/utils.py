# -*- coding: utf-8 -*-

from datetime import datetime
from urllib.parse import urlparse
import os
import logging
import json
logger = logging.getLogger(__name__)

""" 
Utility functions 
"""

def select_random_sublist(l, n):
    random.seed(30)  # use seed to use deterministic randomisation
    assert len(l) >= n, "select_random_sublist: " + str(len(l)) + ' ' + str(n)
    s = random.sample(l, n)
    return s

def run_os_command(cmd):
    ret = subprocess.check_output(cmd, shell=True)
    ret = ret.decode("utf-8").strip()
    return ret

def _sys_command(str_params):
    import subprocess
    result = subprocess.run(str_params, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    return result.returncode, result.stdout, result.stderr


def _pickle_obj_to_file(obj, filePath):
    """ writes obj into a pickle file """
    assert obj is not None
    assert filePath
    # check extension
    assert filePath[-4:] == ".pik"
    pickle.dump(obj, open(filePath, "wb"), pickle.HIGHEST_PROTOCOL)
    log.info("_pickle_obj_to_file: " + filePath)


def _pickle_obj_to_gz(obj, filePath):
    """ writes obj into a gzipped pickle file """
    assert obj is not None
    assert filePath
    # check extension
    assert filePath[-3:] == ".gz"
    pickle.dump(obj, gzip.open(filePath, "wb"), pickle.HIGHEST_PROTOCOL)
    log.info("_pickle_obj_to_gz: " + filePath)

def count_numbers(l):
    vals = [v for v in l if _is_number(v) and not _is_nan(v)]
    n = len(vals)
    return n

def _is_nan(*objs):
    for i in range(len(objs)):
        b = math.isnan(objs[i])
        if not b: return False
    return True

def get_app_settings():
    arr = os.listdir('.')
    with open('mip/app_settings.json') as json_file:
        data = json.load(json_file)

    return data

def _is_number(*objs):
    for i in range(len(objs)):
        b = isinstance(objs[i], (int, float, complex, int32, float64))
        if not b: return False
    return True


def _filter_strings(strArray, patterns, mode):
    """
    """
    assert mode in ['and', 'or']
    assert len(patterns) > 0, "_filter_strings: empty patterns: " + patterns + " mode" + mode
    matches = []
    for s in strArray:
        bAdd = None
        if mode == "and":
            bAdd = True
            for p in patterns:
                if bAdd and re.search(p, s, re.IGNORECASE) is None: bAdd = False
        if mode == "or":
            bAdd = False
            for p in patterns:
                if re.search(p, s, re.IGNORECASE) is not None:
                    bAdd = True
                    continue
        if bAdd == True: matches.append(s)
    return matches


def _write_str_to_file(s, fn, bGzip=False):
    assert _is_str(s, fn)
    if bGzip:
        with gzip.open(fn, "w") as text_file:
            text_file.write(s)
    else:
        with open(fn, "w") as text_file:
            text_file.write(s)
    log.info(str(len(s)) + " chars written in " + fn)

def get_url_domain(url):
    assert url
    dom = urlparse(url).netloc
    return dom

def _wrap_cdata_text(s):
    ss = "<![CDATA[\n" + s + "\n]]>"
    return ss


def _read_str_from_file(fn):
    content = False
    with open(fn) as f:
        content = f.readlines()
    return "".join(content)


def _cut_str(s, maxchar):
    if s is None: return s
    assert _is_str(s)
    if len(s) > maxchar:
        return s[:maxchar] + "..."
    else:
        return s


def strip_none_from_list(l):
    ll = [e for e in l if e is not None]
    return ll


def _get_ellipse_coords(x, y, a, b, angle=0.0, k=2):
    """ Draws an ellipse using (360*k + 1) discrete points; based on pseudo code
    given at http://en.wikipedia.org/wiki/Ellipse
    k = 1 means 361 points (degree by degree)
    a = major axis distance,
    b = minor axis distance,
    x = offset along the x-axis
    y = offset along the y-axis
    angle = clockwise rotation [in degrees] of the ellipse;
        * angle=0  : the ellipse is aligned with the positive x-axis
        * angle=30 : rotated 30 degrees clockwise from positive x-axis
    """
    pts = np.zeros((360 * k + 1, 2))

    beta = -angle * np.pi / 180.0
    sin_beta = np.sin(beta)
    cos_beta = np.cos(beta)
    alpha = np.radians(np.r_[0.:360.:1j * (360 * k + 1)])

    sin_alpha = np.sin(alpha)
    cos_alpha = np.cos(alpha)

    pts[:, 0] = x + (a * cos_alpha * cos_beta - b * sin_alpha * sin_beta)
    pts[:, 1] = y + (a * cos_alpha * sin_beta + b * sin_alpha * cos_beta)

    return pts


def _sort_dict_by_value(d, asc=True):
    """ @return a representation of dictionary d sorted by value """
    s = sorted(d.items(), key=itemgetter(1), reverse=not asc)
    return s


def sort_list_by_attribute(l, attr, asc=True, topK=None):
    """ @return list l sorted by attribute attr """
    assert type(l) is list
    # for ll in l: print "sort_list_by_attribute",type(ll),ll
    sl = sorted(l, key=itemgetter(attr), reverse=not asc)
    if topK is not None:
        assert not asc
        return sl[:topK]
    return sl


def _clean_str_for_xml(s):
    clean_s = ''.join(c for c in s if _valid_XML_char_ordinal(ord(c)))
    # print clean_s
    clean_s = clean_s.decode('utf-8')
    # print clean_s
    return clean_s


def _str_to_ascii(a):
    assert _is_str(a)
    return a.decode('ascii', 'ignore')


def _split_list(alist, wanted_parts):
    length = len(alist)
    sublists = [alist[i * length // wanted_parts: (i + 1) * length // wanted_parts]
                for i in range(wanted_parts)]
    i = 0
    for s in sublists: i += len(s)
    assert i == len(alist)
    return sublists


def _parallel_for(input_array, n_threads, funct):
    print("_parallel_for threads=",n_threads,"input n=",len(input_array))
    sw = StopWatch("_parallel_for thrd ="+str(n_threads)+" input_sz ="+str(len(input_array)))
    from joblib import Parallel, delayed
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    assert n_threads >= 1 and n_threads <= num_cores
    # call (SLOW)
    results = Parallel(n_jobs=n_threads, prefer="threads")(delayed(funct)(inobj) for inobj in input_array)
    sw.tick()
    assert len(results) == len(input_array), "_parallel_for task failed"
    return results

def move_col_front_df(df, tcol):
    assert tcol in df.columns
    df1 = df[ [tcol] + [ col for col in df.columns if col != tcol ] ]
    return df1


def get_current_thread_id():
    id = str(threading.current_thread().name +':'+ str(threading.current_thread().ident) )
    return id


def scale_array(values):
    """ Scale array between 0 and 1 """
    values = np.array(values)
    scaledV = 1 - (values / float(max(values)))
    return scaledV.tolist()


def make_dir(fdir, delete_contents=False):
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    else:
        if delete_contents:
            shutil.rmtree(fdir)


def scale_array_max(values, maxVal):
    """ Scale array between 0 and 1 """
    values = np.array(values)
    scaledV = 1 - (values / float(maxVal))
    return scaledV.tolist()


def _float_eq(a, b, err=1e-08):
    return abs(a - b) <= err


def memory_usage():
    # return the memory usage in bytes
    process = psutil.Process(os.getpid())
    # mem = process.get_memory_info()[0] / float(2 ** 20)
    mem = process.get_memory_info()[0]  # / float(2 ** 20)
    return mem


def memory_usage_str():
    return ">> memory_usage: " + strmem(memory_usage())


def strh(n):
    """ to string human (generic number)"""
    assert _is_number(n)
    s = humanize.intcomma(n)
    return s


def strmem(n):
    """ to string human (memory) """
    assert _is_number(n)
    s = humanize.naturalsize(n, gnu=True)
    return s

class StopWatch(object):
    """
    StopWatch util 
    """
    def __init__(self, desc):
        self._tstart = datetime.now()
        #self._tstart = time.process_time()
        # time.process_time()
        self._tcur = self._tstart
        self._desc = desc

    def tick(self, desc=''):
        t = datetime.now() - self._tcur
        #t = time.process_time() - self._tcur
        # tt = datetime.now() - self._tstart
        bFirst = self._tcur == self._tstart
        self._tcur = datetime.now()
        msg = "> SW [" + self._desc + " " + desc + "] t=" + str(t)[:-3]
        return msg

