class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def print_metrics(matrix):
    precision = matrix['TP']/(matrix['TP']+matrix['FP'])
    recall = matrix['TP']/(matrix['TP']+matrix['FN'])
    acc = (matrix['TP']+matrix['TN']) / \
        (matrix['TP']+matrix['FN']+matrix['FP']+matrix['TN'])
    f1 = (2*precision*recall)/(precision+recall)
    print('TP: {}, TN: {}, FP: {}, FN: {}'.format(
        matrix['TP'], matrix['TN'], matrix['FP'], matrix['FN']))
    print('acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f} \n'.format(
        acc, precision, recall, f1))
    return 0

def get_mertics(matrix):
    precision = matrix['TP']/(matrix['TP']+matrix['FP'])
    recall = matrix['TP']/(matrix['TP']+matrix['FN'])
    acc = (matrix['TP']+matrix['TN']) / \
        (matrix['TP']+matrix['FN']+matrix['FP']+matrix['TN'])
    f1 = (2*precision*recall)/(precision+recall)
    spe = matrix['TN']/(matrix['TN']+matrix['FP'])
    return acc,precision,recall,f1,spe

def list_to_tuple(seq_list):
    result=[]
    for seq in seq_list:
        item=('1',seq)
        result.append(item)
    return result