# -*- coding: utf-8 -*-

""" 
VPN functions 

https://www.privateinternetaccess.com/helpdesk/kb/articles/pia-desktop-command-line-interface

"""

vpn_servers = ['uk-manchester','uk-london','uk-southampton','ireland','belgium',
                'isle-of-man','luxembourg','austria']

vpn_os = 'mac' # 'win'

assert VPN_OS in ['mac','win']

# %% Mac functions
def vpn_off():
    if vpn_os == 'mac':
        ret = run_os_command('piactl disconnect')
        print(ret)

def vpn_on():
    if vpn_os == 'mac':
        ret = run_os_command('piactl connect')
        print(ret)

def vpn_go_region(reg):
    assert reg in VPN_SERVERS
    if vpn_os == 'mac':
        ret = run_os_command('piactl set region '+reg)
    print(">> vpn_go_region:",reg)
    return ret

def vpn_is_on():
    if vpn_os == 'mac':
        ret = run_os_command('piactl get connectionstate') == 'Connected'
        return ret

# %% Windows functions
def vpn_random_region():
    vpn_go_region(random.choice(vpn_servers))

def run_os_command(cmd):
    if vpn_os == 'mac':
        ret = subprocess.check_output(cmd, shell=True)
        ret = ret.decode("utf-8").strip()
        return ret


# TODO: add commands for PIA on Windows

