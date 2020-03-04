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
        config['num_epoch'] = int(train[0].getElementsByTagName('num_epoch')[0].firstChild.data)
        config['stage_two_three'] = int(train[0].getElementsByTagName('stage_two_three')[0].firstChild.data)
        config['repeat_init_block'] = int(train[0].getElementsByTagName('repeat_init_block')[0].firstChild.data)
        skip = str(train[0].getElementsByTagName('skip')[0].firstChild.data)
        config['skip'] = False
        if skip == 'True':
            config['skip'] = True

        optimize = collection.getElementsByTagName('optimize')
        config['learning_rate'] = float(optimize[0].getElementsByTagName('learning_rate')[0].firstChild.data)
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

        return config


