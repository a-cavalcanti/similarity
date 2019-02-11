import os
import xml.etree.ElementTree as etree

def readDataSet(arq):
    # Get the path to the directory where the project is located
    BASE_PATH = os.path.dirname(os.path.realpath(__file__))

    xml_file = os.path.join(BASE_PATH, arq)

    print(xml_file)

    # Access the XML file, save it to memory, and parse it
    tree = etree.parse(xml_file)

    # Get root element with .getroot()
    root = tree.getroot()

    # Loop through the elements
    # Use the .tag and .attrib properties
    # for child in root:
    #     print(child.tag, child.attrib)

    dataSet = []
    # Use the .text property to view the data inside of the elements
    for child in root:
        line = []
        line.append(str(child.attrib['id']))
        line.append(child.attrib['entailment'])
        line.append(child.attrib['similarity'])

        for element in child:
            #print(element.tag, ":", element.text)
            line.append(element.text)
        dataSet.append(line)

    return dataSet