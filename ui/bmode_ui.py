# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'bmode_v2.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
    QLineEdit, QPlainTextEdit, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(600, 400)
        self.horizontalLayout_3 = QHBoxLayout(Form)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_bmode = QHBoxLayout()
        self.horizontalLayout_bmode.setObjectName(u"horizontalLayout_bmode")
        self.verticalLayout_bmode_menu = QVBoxLayout()
        self.verticalLayout_bmode_menu.setObjectName(u"verticalLayout_bmode_menu")
        self.comboBox_bmode_streamOption = QComboBox(Form)
        self.comboBox_bmode_streamOption.addItem("")
        self.comboBox_bmode_streamOption.addItem("")
        self.comboBox_bmode_streamOption.addItem("")
        self.comboBox_bmode_streamOption.setObjectName(u"comboBox_bmode_streamOption")

        self.verticalLayout_bmode_menu.addWidget(self.comboBox_bmode_streamOption)

        self.comboBox_bmode_streamPort = QComboBox(Form)
        self.comboBox_bmode_streamPort.setObjectName(u"comboBox_bmode_streamPort")
        self.comboBox_bmode_streamPort.setEnabled(True)

        self.verticalLayout_bmode_menu.addWidget(self.comboBox_bmode_streamPort)

        self.horizontalLayout_bmode_calib = QHBoxLayout()
        self.horizontalLayout_bmode_calib.setObjectName(u"horizontalLayout_bmode_calib")
        self.lineEdit_bmode_calibPath = QLineEdit(Form)
        self.lineEdit_bmode_calibPath.setObjectName(u"lineEdit_bmode_calibPath")
        self.lineEdit_bmode_calibPath.setEnabled(True)

        self.horizontalLayout_bmode_calib.addWidget(self.lineEdit_bmode_calibPath)

        self.pushButton_bmode_calibClear = QPushButton(Form)
        self.pushButton_bmode_calibClear.setObjectName(u"pushButton_bmode_calibClear")
        icon = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.EditClear))
        self.pushButton_bmode_calibClear.setIcon(icon)

        self.horizontalLayout_bmode_calib.addWidget(self.pushButton_bmode_calibClear)

        self.pushButton_bmode_calibBrowse = QPushButton(Form)
        self.pushButton_bmode_calibBrowse.setObjectName(u"pushButton_bmode_calibBrowse")
        self.pushButton_bmode_calibBrowse.setEnabled(True)
        icon1 = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.DocumentOpen))
        self.pushButton_bmode_calibBrowse.setIcon(icon1)

        self.horizontalLayout_bmode_calib.addWidget(self.pushButton_bmode_calibBrowse)


        self.verticalLayout_bmode_menu.addLayout(self.horizontalLayout_bmode_calib)

        self.pushButton_bmode_openStream = QPushButton(Form)
        self.pushButton_bmode_openStream.setObjectName(u"pushButton_bmode_openStream")
        self.pushButton_bmode_openStream.setEnabled(True)

        self.verticalLayout_bmode_menu.addWidget(self.pushButton_bmode_openStream)

        self.plainTextEdit_bmode_textStream = QPlainTextEdit(Form)
        self.plainTextEdit_bmode_textStream.setObjectName(u"plainTextEdit_bmode_textStream")

        self.verticalLayout_bmode_menu.addWidget(self.plainTextEdit_bmode_textStream)

        self.horizontalLayout_bmode_recorddir = QHBoxLayout()
        self.horizontalLayout_bmode_recorddir.setObjectName(u"horizontalLayout_bmode_recorddir")
        self.lineEdit_bmode_recorddir = QLineEdit(Form)
        self.lineEdit_bmode_recorddir.setObjectName(u"lineEdit_bmode_recorddir")

        self.horizontalLayout_bmode_recorddir.addWidget(self.lineEdit_bmode_recorddir)

        self.pushButton_bmode_recorddirClear = QPushButton(Form)
        self.pushButton_bmode_recorddirClear.setObjectName(u"pushButton_bmode_recorddirClear")
        self.pushButton_bmode_recorddirClear.setIcon(icon)

        self.horizontalLayout_bmode_recorddir.addWidget(self.pushButton_bmode_recorddirClear)

        self.pushButton_bmode_recorddirBrowse = QPushButton(Form)
        self.pushButton_bmode_recorddirBrowse.setObjectName(u"pushButton_bmode_recorddirBrowse")
        icon2 = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.FolderOpen))
        self.pushButton_bmode_recorddirBrowse.setIcon(icon2)

        self.horizontalLayout_bmode_recorddir.addWidget(self.pushButton_bmode_recorddirBrowse)


        self.verticalLayout_bmode_menu.addLayout(self.horizontalLayout_bmode_recorddir)

        self.pushButton_bmode_recordStream = QPushButton(Form)
        self.pushButton_bmode_recordStream.setObjectName(u"pushButton_bmode_recordStream")

        self.verticalLayout_bmode_menu.addWidget(self.pushButton_bmode_recordStream)


        self.horizontalLayout_bmode.addLayout(self.verticalLayout_bmode_menu)

        self.label_bmode_image = QLabel(Form)
        self.label_bmode_image.setObjectName(u"label_bmode_image")
        self.label_bmode_image.setAutoFillBackground(False)
        self.label_bmode_image.setStyleSheet(u"background-color: rgb(0,0,0);")
        self.label_bmode_image.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_bmode.addWidget(self.label_bmode_image)

        self.horizontalLayout_bmode.setStretch(0, 1)
        self.horizontalLayout_bmode.setStretch(1, 3)

        self.horizontalLayout_3.addLayout(self.horizontalLayout_bmode)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.comboBox_bmode_streamOption.setItemText(0, QCoreApplication.translate("Form", u"Stream Image", None))
        self.comboBox_bmode_streamOption.setItemText(1, QCoreApplication.translate("Form", u"Stream Screen (Other)", None))
        self.comboBox_bmode_streamOption.setItemText(2, QCoreApplication.translate("Form", u"Stream Screen (This PC)", None))

        self.comboBox_bmode_streamPort.setPlaceholderText(QCoreApplication.translate("Form", u"Select USB Port", None))
        self.lineEdit_bmode_calibPath.setPlaceholderText(QCoreApplication.translate("Form", u"~/config.xml", None))
        self.pushButton_bmode_calibClear.setText("")
        self.pushButton_bmode_calibBrowse.setText(QCoreApplication.translate("Form", u"Calib File", None))
        self.pushButton_bmode_openStream.setText(QCoreApplication.translate("Form", u"Open Stream", None))
        self.lineEdit_bmode_recorddir.setPlaceholderText(QCoreApplication.translate("Form", u"~/output/", None))
        self.pushButton_bmode_recorddirClear.setText("")
        self.pushButton_bmode_recorddirBrowse.setText(QCoreApplication.translate("Form", u"Record Dir", None))
        self.pushButton_bmode_recordStream.setText(QCoreApplication.translate("Form", u"Record", None))
        self.label_bmode_image.setText("")
    # retranslateUi

