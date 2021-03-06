import xml.dom.minidom as minidom

class config(object):
    def __init__(self):
        return

    @staticmethod
    def get_config(xml_path):
        dom_tree = minidom.parse(xml_path)
        collection = dom_tree.documentElement
        if collection.nodeName != 'config':
            raise RuntimeError('this is invalid nn config: the must has header "config"')

        config = dict()

        train = collection.getElementsByTagName('train')
        config['batch_size'] = int(train[0].getElementsByTagName('batch_size')[0].firstChild.data)
        config['eval_batch_size'] = int(train[0].getElementsByTagName('eval_batch_size')[0].firstChild.data)
        config['train_epoch'] = int(train[0].getElementsByTagName('train_epoch')[0].firstChild.data)
        config['stage_two_three'] = int(train[0].getElementsByTagName('stage_two_three')[0].firstChild.data)
        skip = str(train[0].getElementsByTagName('skip')[0].firstChild.data)
        config['skip'] = False
        if skip == 'True':
            config['skip'] = True
        config['device'] = str(train[0].getElementsByTagName('device')[0].firstChild.data)
        if config['device'] == 'None':
            config['device'] = None
        device_log = str(train[0].getElementsByTagName('device_log')[0].firstChild.data)
        config['device_log'] = False
        if device_log == 'True':
            config['device_log'] = True

        optimize = collection.getElementsByTagName('optimize')
        config['learning_rate'] = float(optimize[0].getElementsByTagName('learning_rate')[0].firstChild.data)
        config['end_learning_rate'] = float(optimize[0].getElementsByTagName('end_learning_rate')[0].firstChild.data)
        config['decay_rate'] = float(optimize[0].getElementsByTagName('decay_rate')[0].firstChild.data)
        config['epsilon'] = float(optimize[0].getElementsByTagName('epsilon')[0].firstChild.data)
        config['num_epochs_before_decay'] = float(optimize[0].getElementsByTagName('num_epochs_before_decay')[0].firstChild.data)
        config['l2_weight_decay'] = float(optimize[0].getElementsByTagName('l2_weight_decay')[0].firstChild.data)

        mode = collection.getElementsByTagName('mode')
        config['mode_path'] = mode[0].getElementsByTagName('mode_path')[0].firstChild.data
        config['update_mode_freq'] = int(mode[0].getElementsByTagName('update_mode_freq')[0].firstChild.data)

        image = collection.getElementsByTagName('image')
        config['image_path'] = image[0].getElementsByTagName('image_path')[0].firstChild.data
        config['image_suffix'] = image[0].getElementsByTagName('image_suffix')[0].firstChild.data
        config['class_num'] = int(image[0].getElementsByTagName('class_num')[0].firstChild.data)
        config['result_path'] = image[0].getElementsByTagName('result_path')[0].firstChild.data
        config['img_width'] = int(image[0].getElementsByTagName('img_width')[0].firstChild.data)
        config['img_height'] = int(image[0].getElementsByTagName('img_height')[0].firstChild.data)

        log_path = collection.getElementsByTagName('log_path')
        config['log_path'] = log_path[0].firstChild.data
        return config


