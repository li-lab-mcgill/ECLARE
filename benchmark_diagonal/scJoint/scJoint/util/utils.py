import sys
import time
import os
import torch
import shutil

last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    """Progress Bar for display."""
    def _format_time(seconds):
        days = int(seconds / 3600 / 24)
        seconds = seconds - days * 3600 * 24
        hours = int(seconds / 3600)
        seconds = seconds - hours * 3600
        minutes = int(seconds / 60)
        seconds = seconds - minutes * 60
        secondsf = int(seconds)
        seconds = seconds - secondsf
        millis = int(seconds * 1000)

        f = ''
        i = 1
        if days > 0:
            f += str(days) + 'D'
            i += 1
        if hours > 0 and i <= 2:
            f += str(hours) + 'h'
            i += 1
        if minutes > 0 and i <= 2:
            f += str(minutes) + 'm'
            i += 1
        if secondsf > 0 and i <= 2:
            f += str(secondsf) + 's'
            i += 1
        if millis > 0 and i <= 2:
            f += str(millis) + 'ms'
            i += 1
        if f == '':
            f = '0ms'
        return f

    # Fallback for terminal width
    try:
        term_size = os.popen('stty size 2>/dev/null', 'r').read().split()
        if len(term_size) == 2:
            _, term_width = term_size
            term_width = int(term_width)  # Convert to integer
        else:
            raise ValueError("Invalid output from 'stty size'")
    except (ValueError, OSError):
        term_width = 80  # Default width
    
    term_width = int(term_width)
    TOTAL_BAR_LENGTH = 30.0
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.
        last_time = begin_time

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    sys.stdout.write('=' * cur_len)
    sys.stdout.write('>')
    sys.stdout.write('.' * rest_len)
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('    Step: %s' % _format_time(step_time))
    L.append(' | Tot: %s' % _format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    spaces = term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3
    sys.stdout.write(' ' * max(spaces, 0))

    # Go back to the center of the bar.
    sys.stdout.write('\b' * (term_width - int(TOTAL_BAR_LENGTH / 2)))
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


# refer to https://github.com/xternalz/WideResNet-pytorch
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "models/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
