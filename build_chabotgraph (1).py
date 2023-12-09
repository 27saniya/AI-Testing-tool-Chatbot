#!/usr/bin/env python3
# coding: utf-8
# Date: 21-02-01

import os
import json
from py2neo import Graph,Node

class ChatbotGraph:
    def __init__(self):
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.data_path = os.path.join(cur_dir, 'ai_yash.json')
        self.g = Graph("bolt://localhost:7687", user="neo4j", password="ABCD1234")

    '''read file'''
    def read_nodes(self):
        # 4 nodes
        tools = [] 
        objects = [] 
        operations = [] 
        pseudo_names = []

        operation_infos = [] #information of operations

        # relationships
        rels_object = [] # operation & subjects
        rels_tool = [] # operation & terminology
        rels_pseudo_name = [] # operation & alias


        count = 0
        for data in open(self.data_path, encoding="utf8"):
            operation_dict = {}
            count += 1
            print(count)
            data_json = json.loads(data)
            operation = data_json['name']
            operation_dict['name'] = operation
            operations.append(operation)
            operation_dict['description'] = ''


            if 'pseudo_name' in data_json:
                pseudo_names += data_json['pseudo_name']
                for pseudo_name in data_json['pseudo_name']:
                    rels_pseudo_name.append([operation, pseudo_name])

            if 'object' in data_json:
                object = data_json['object']
                for _object in object:
                    rels_object.append([operation, _object])
                operation_dict['object'] = object
                objects += object

            if 'description' in data_json:
                operation_dict['description'] = data_json['description']

            if 'tool' in data_json:
                tool = data_json['tool']
                for _tool in tool:
                    rels_tool.append([operation, _tool])
                tools += tool

            operation_infos.append(operation_dict)
        return set(tools), set(objects), set(pseudo_names), set(operations), operation_infos,rels_pseudo_name,rels_object,rels_tool

    '''create node'''
    def create_node(self, label, nodes):
        count = 0
        for node_name in nodes:
            node = Node(label, name=node_name)
            self.g.create(node)
            count += 1
            print(count, len(nodes))
        return

    '''create central point'''
    def create_operation_nodes(self, operation_infos):
        count = 0
        for operation_dict in operation_infos:
            node = Node("Operation", name=operation_dict['name'], method=operation_dict['description'])
            self.g.create(node)
            count += 1
            print(count)
        return

    '''create kg schema'''
    def create_graphnodes(self):
        Tools, Objects, Pseudo_names, Operations, operation_infos,rels_pseudo_name, rels_object,rels_tool = self.read_nodes()
        self.create_operation_nodes(operation_infos)
        self.create_node('Tool', Tools)
        print(Tools)
        self.create_node('Object', Objects)
        print(Objects)
        self.create_node('Pseudo_name', Pseudo_names)
        print(Pseudo_names)
        return


    '''create entities'''
    def create_graphrels(self):
        Tools, Objects, Pseudo_names, Operations, operation_infos,rels_pseudo_name, rels_object,rels_tool = self.read_nodes()
        self.create_relationship('Operation', 'Tool', rels_tool, 'contain', 'contain')
        self.create_relationship('Operation', 'Pseudo_name', rels_pseudo_name, 'has_pseudo_name', 'has name of')
        self.create_relationship('Operation', 'Object', rels_object, 'belongs_to', 'belong to')

    '''create relationship'''
    def create_relationship(self, start_node, end_node, edges, rel_type, rel_name):
        count = 0
        set_edges = []
        for edge in edges:
            set_edges.append('###'.join(edge))
        all = len(set(set_edges))
        for edge in set(set_edges):
            edge = edge.split('###')
            p = edge[0]
            q = edge[1]
            query = "match(p:%s),(q:%s) where p.name='%s'and q.name='%s' create (p)-[rel:%s{name:'%s'}]->(q)" % (
                start_node, end_node, p, q, rel_type, rel_name)
            try:
                self.g.run(query)
                count += 1
                print(rel_type, count, all)
            except Exception as e:
                print(e)
        return

    '''export data'''
    def export_data(self):
        Tools, Objects, Pseudo_names, Operations, operation_infos,rels_pseudo_name, rels_object,rels_tool = self.read_nodes()
        f_tool = open('tool.txt', 'w+')
        f_object = open('object.txt', 'w+')
        f_pseudo_name = open('pseudo_name.txt', 'w+')
        f_operation = open('operation.txt', 'w+')


        f_tool.write('\n'.join(list(Tools)))
        f_object.write('\n'.join(list(Objects)))
        f_pseudo_name.write('\n'.join(list(Pseudo_names)))
        f_operation.write('\n'.join(list(Operations)))


        f_tool.close()
        f_object.close()
        f_pseudo_name.close()
        f_operation.close()

        return



if __name__ == '__main__':
    handler = ChatbotGraph()
    print("step1:loading entities")
    handler.create_graphnodes()
    print("step2:loading relationship")      
    handler.create_graphrels()
    handler.export_data()
    
