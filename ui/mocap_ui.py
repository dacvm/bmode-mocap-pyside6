# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mocap_v2.ui'
##
## Created by: Qt User Interface Compiler version 6.10.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLineEdit,
    QPlainTextEdit, QPushButton, QSizePolicy, QVBoxLayout,
    QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(600, 400)
        self.horizontalLayout_2 = QHBoxLayout(Form)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_mocap = QHBoxLayout()
        self.horizontalLayout_mocap.setObjectName(u"horizontalLayout_mocap")
        self.verticalLayout_mocap_menu = QVBoxLayout()
        self.verticalLayout_mocap_menu.setObjectName(u"verticalLayout_mocap_menu")
        self.comboBox_mocap_systemSelect = QComboBox(Form)
        self.comboBox_mocap_systemSelect.addItem("")
        self.comboBox_mocap_systemSelect.addItem("")
        self.comboBox_mocap_systemSelect.setObjectName(u"comboBox_mocap_systemSelect")

        self.verticalLayout_mocap_menu.addWidget(self.comboBox_mocap_systemSelect)

        self.horizontalLayout_mocap_ip = QHBoxLayout()
        self.horizontalLayout_mocap_ip.setObjectName(u"horizontalLayout_mocap_ip")
        self.lineEdit_mocap_ip = QLineEdit(Form)
        self.lineEdit_mocap_ip.setObjectName(u"lineEdit_mocap_ip")

        self.horizontalLayout_mocap_ip.addWidget(self.lineEdit_mocap_ip)

        self.pushButton_mocap_openStream = QPushButton(Form)
        self.pushButton_mocap_openStream.setObjectName(u"pushButton_mocap_openStream")

        self.horizontalLayout_mocap_ip.addWidget(self.pushButton_mocap_openStream)


        self.verticalLayout_mocap_menu.addLayout(self.horizontalLayout_mocap_ip)

        self.plainTextEdit_mocap_textStream = QPlainTextEdit(Form)
        self.plainTextEdit_mocap_textStream.setObjectName(u"plainTextEdit_mocap_textStream")
        self.plainTextEdit_mocap_textStream.setEnabled(True)
        font = QFont()
        font.setFamilies([u"Courier New"])
        font.setPointSize(7)
        self.plainTextEdit_mocap_textStream.setFont(font)

        self.verticalLayout_mocap_menu.addWidget(self.plainTextEdit_mocap_textStream)

        self.horizontalLayout_mocap_recorddir = QHBoxLayout()
        self.horizontalLayout_mocap_recorddir.setObjectName(u"horizontalLayout_mocap_recorddir")
        self.lineEdit_mocap_recorddir = QLineEdit(Form)
        self.lineEdit_mocap_recorddir.setObjectName(u"lineEdit_mocap_recorddir")

        self.horizontalLayout_mocap_recorddir.addWidget(self.lineEdit_mocap_recorddir)

        self.pushButton_mocap_recorddirClear = QPushButton(Form)
        self.pushButton_mocap_recorddirClear.setObjectName(u"pushButton_mocap_recorddirClear")
        icon = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.EditClear))
        self.pushButton_mocap_recorddirClear.setIcon(icon)

        self.horizontalLayout_mocap_recorddir.addWidget(self.pushButton_mocap_recorddirClear)

        self.pushButton_mocap_recorddirBrowse = QPushButton(Form)
        self.pushButton_mocap_recorddirBrowse.setObjectName(u"pushButton_mocap_recorddirBrowse")
        icon1 = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.FolderOpen))
        self.pushButton_mocap_recorddirBrowse.setIcon(icon1)

        self.horizontalLayout_mocap_recorddir.addWidget(self.pushButton_mocap_recorddirBrowse)


        self.verticalLayout_mocap_menu.addLayout(self.horizontalLayout_mocap_recorddir)

        self.pushButton_mocap_record = QPushButton(Form)
        self.pushButton_mocap_record.setObjectName(u"pushButton_mocap_record")

        self.verticalLayout_mocap_menu.addWidget(self.pushButton_mocap_record)


        self.horizontalLayout_mocap.addLayout(self.verticalLayout_mocap_menu)

        self.widget_mocap_matplotlib = QWidget(Form)
        self.widget_mocap_matplotlib.setObjectName(u"widget_mocap_matplotlib")
        self.verticalLayout_2 = QVBoxLayout(self.widget_mocap_matplotlib)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_mocap_matplotlib = QVBoxLayout()
        self.verticalLayout_mocap_matplotlib.setObjectName(u"verticalLayout_mocap_matplotlib")

        self.verticalLayout_2.addLayout(self.verticalLayout_mocap_matplotlib)


        self.horizontalLayout_mocap.addWidget(self.widget_mocap_matplotlib)

        self.horizontalLayout_mocap.setStretch(0, 1)
        self.horizontalLayout_mocap.setStretch(1, 3)

        self.horizontalLayout_2.addLayout(self.horizontalLayout_mocap)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.comboBox_mocap_systemSelect.setItemText(0, QCoreApplication.translate("Form", u"Qualisys", None))
        self.comboBox_mocap_systemSelect.setItemText(1, QCoreApplication.translate("Form", u"Vicon", None))

        self.lineEdit_mocap_ip.setPlaceholderText(QCoreApplication.translate("Form", u"127.0.0.1", None))
        self.pushButton_mocap_openStream.setText(QCoreApplication.translate("Form", u"Open Stream", None))
        self.lineEdit_mocap_recorddir.setPlaceholderText(QCoreApplication.translate("Form", u"~/output/", None))
        self.pushButton_mocap_recorddirClear.setText("")
        self.pushButton_mocap_recorddirBrowse.setText(QCoreApplication.translate("Form", u"Record Dir", None))
        self.pushButton_mocap_record.setText(QCoreApplication.translate("Form", u"Record", None))
    # retranslateUi

