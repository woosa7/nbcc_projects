#!/home/nbcc/anaconda3/envs/prowave/bin/python
import os
import argparse
import subprocess
import smtplib
from email.mime.text import MIMEText
from datetime import datetime


PARSER = argparse.ArgumentParser()
PARSER.add_argument('work_id')
PARSER.add_argument('recipient')
PARSER.add_argument('--batch', action='store_true')
PARSER.add_argument('--dependency')


def submit_batch(work_id, recipient, dependency):
    cmd = [
        '/usr/bin/sbatch',
        '--nodes', '1',
        '--time', '168:00:00',
        '--job-name', 'MAIL',
        '--ntasks', '1',
        '--output', '/dev/null',
        '--error', '/dev/null',
    ]

    if dependency:
        cmd += ['--dependency', dependency]

    cmd += [
        os.path.abspath(__file__),
        work_id,
        recipient,
    ]

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()

        submit_msg = out.decode()
        if submit_msg.startswith('Submitted batch job'):
            batch_jobid = int(submit_msg.split()[3])
            # with open(os.path.join(prep_dir, 'status'), 'w') as f:
            #     f.write('submitted %d' % batch_jobid)
            return batch_jobid
        else:
            return 'Failed'
    except FileNotFoundError:
        return '-1'


if __name__ == '__main__':
    args = PARSER.parse_args()

    if args.batch:
        submit_batch(args.work_id, args.recipient, args.dependency)
    else:
        print('sending mail...')
        try:
            smtp = smtplib.SMTP('smtp.gmail.com', 587)
            smtp.ehlo()
            smtp.starttls()
            smtp.login(os.environ['EMAIL_HOST_USER'], os.environ['EMAIL_HOST_PASSWORD'])

            msg = MIMEText("""
            Your job is done at %s UTC.
            If you want to check your calculation result, please visit following link.
            
            https://www.prowave.org/solvation-free-energy/%s/
            
            If you have any message to us, feel free to reply this mail.
            Thank you!
            """ % (datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'), args.work_id))

            msg['Subject'] = '[ProWaVE] Your job is done!'
            msg['To'] = args.recipient
            smtp.sendmail(os.environ['EMAIL_HOST_USER'], msg['To'], msg.as_string())
        except:
            import traceback
            print('sending mail failed')
            prowave_data_path = os.environ.get('PROWAVE_DATA_DIR', 'prowave_data')
            with open(os.path.join(prowave_data_path, '%s/send_mail_err' % args.work_id), 'w') as f:
                print(traceback.format_exc(), file=f)
