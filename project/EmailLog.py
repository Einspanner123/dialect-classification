import time
import os
import smtplib
from email.mime.text import MIMEText
from email.header import Header


class EmailLog:
    def __init__(self):
        time_info = str(time.strftime("%Y/%m/%d", time.localtime()))
        if not os.path.exists('./runlogs/'):
            os.mkdir('./runlogs/')
        self.file_name = './runlogs/' + time_info + '.log'
        self.file = open(self.file_name, 'a')

    def __del__(self):
        self.file.close()

    # 添加日志记录
    def add_log(self, log):
        time_info = str(time.strftime("%H:%M:%S\n", time.localtime()))
        self.file.write(time_info + log)
        self.file.write('\r')
        self.file.flush()

    # 运行结束后发送邮件
    def send_mail(self):
        self.file.close()
        from_addr = '767781336@qq.com'  # 邮件发送账号
        to_addr = 'linxintao_fm@foxmail.com'  # 接收邮件账号
        code = 'dykhvgmslynxbdbj'  # 授权码
        smtp_server = 'smtp.qq.com'
        smtp_port = 465
        # 配置服务器
        stmp = smtplib.SMTP_SSL(smtp_server, smtp_port)
        stmp.login(from_addr, code)
        with open(self.file_name, 'r') as f:
            buffer = f.read()
        # 组装发送内容
        message = MIMEText(buffer, 'plain', 'utf-8')  # 发送的内容
        message['From'] = Header("autodl", 'utf-8')  # 发件人
        message['To'] = Header("me", 'utf-8')  # 收件人
        subject = 'AUTODL: 运行结束'
        message['Subject'] = Header(subject, 'utf-8')  # 邮件标题
        try:
            stmp.sendmail(from_addr, to_addr, message.as_string())
            print('Sending e-mail succeeded.')
        except Exception as e:
            print('Sending e-mail failed:\n' + str(e))
