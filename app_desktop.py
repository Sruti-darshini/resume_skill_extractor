import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QFileDialog, QWidget, QScrollArea, QFrame, QSizePolicy, QLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QRect, QPoint
from PyQt5.QtGui import QPixmap, QFont, QIcon, QColor, QPalette

# Import your existing processing functions
from resume_core import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_skills,
    extract_name,
    extract_email,
    extract_phone,
)

class WorkerThread(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
    
    def run(self):
        try:
            # Extract text based on file type
            if self.file_path.lower().endswith('.pdf'):
                text = extract_text_from_pdf(self.file_path)
            elif self.file_path.lower().endswith(('.doc', '.docx')):
                text = extract_text_from_docx(self.file_path)
            else:
                self.error.emit("Unsupported file format")
                return
                
            # Extract information
            result = {
                'name': extract_name(text),
                'email': extract_email(text),
                'phone': extract_phone(text),
                'skills': extract_skills(text)
            }
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))

class ResumeSkillExtractorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Resume Skill Extractor")
        self.setMinimumSize(800, 700)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f7fa;
            }
            QLabel#title {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                margin: 20px 0;
            }
            QPushButton {
                background-color: #4361ee;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #3a56d4;
            }
            QPushButton:disabled {
                background-color: #b8c2ec;
            }
            QFrame#resultFrame {
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                margin-top: 20px;
            }
            QLabel#sectionTitle {
                font-size: 18px;
                font-weight: bold;
                color: #2c3e50;
                margin: 15px 0 10px 0;
            }
            QLabel#skillTag {
                background-color: #e0e7ff;
                color: #3b82f6;
                padding: 5px 10px;
                border-radius: 15px;
                margin: 5px;
                display: inline-block;
            }
            QLabel#infoText {
                font-size: 14px;
                color: #4b5563;
                margin: 5px 0;
            }
        """)
        
        self.init_ui()
        
    def init_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setAlignment(Qt.AlignTop)
        
        # Title
        title = QLabel("Resume Skill Extractor")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Upload section
        upload_layout = QVBoxLayout()
        upload_layout.setAlignment(Qt.AlignCenter)
        
        # Upload button
        self.upload_btn = QPushButton("Select Resume")
        self.upload_btn.setIcon(self.style().standardIcon(getattr(self.style(), 'SP_FileIcon')))
        self.upload_btn.clicked.connect(self.select_file)
        
        # File info label
        self.file_info = QLabel("No file selected")
        self.file_info.setAlignment(Qt.AlignCenter)
        self.file_info.setObjectName("infoText")
        
        # Parse button
        self.parse_btn = QPushButton("Parse Resume")
        self.parse_btn.setEnabled(False)
        self.parse_btn.clicked.connect(self.parse_resume)
        
        upload_layout.addWidget(self.upload_btn)
        upload_layout.addWidget(self.file_info)
        upload_layout.addWidget(self.parse_btn)
        
        # Add upload section to main layout
        layout.addLayout(upload_layout)
        
        # Results section
        self.result_frame = QFrame()
        self.result_frame.setObjectName("resultFrame")
        self.result_frame.setVisible(False)
        result_layout = QVBoxLayout(self.result_frame)
        
        # Scroll area for results
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_content)
        scroll.setWidget(scroll_content)
        
        result_layout.addWidget(scroll)
        layout.addWidget(self.result_frame)
        
        # Status bar
        self.statusBar = self.statusBar()
        
        # Worker thread
        self.worker_thread = None
        
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Resume",
            "",
            "Resume Files (*.pdf *.docx *.doc)"
        )
        
        if file_path:
            self.file_path = file_path
            self.file_info.setText(os.path.basename(file_path))
            self.parse_btn.setEnabled(True)
    
    def parse_resume(self):
        if not hasattr(self, 'file_path'):
            return
            
        # Clear previous results
        self.clear_results()
        self.statusBar.showMessage("Processing resume...")
        
        # Disable buttons during processing
        self.upload_btn.setEnabled(False)
        self.parse_btn.setEnabled(False)
        self.parse_btn.setText("Processing...")
        
        # Create and start worker thread
        self.worker_thread = WorkerThread(self.file_path)
        self.worker_thread.finished.connect(self.show_results)
        self.worker_thread.error.connect(self.show_error)
        self.worker_thread.finished.connect(self.worker_finished)
        self.worker_thread.error.connect(self.worker_finished)
        self.worker_thread.start()
    
    def clear_results(self):
        # Clear previous results
        for i in reversed(range(self.scroll_layout.count())): 
            self.scroll_layout.itemAt(i).widget().setParent(None)
    
    def show_results(self, result):
        self.result_frame.setVisible(True)
        
        # Add name
        if result['name']:
            self.add_section("Name", result['name'])
        
        # Add contact info
        contact_info = []
        if result['email']:
            contact_info.append(f"📧 {result['email']}")
        if result['phone']:
            contact_info.append(f"📞 {result['phone']}")
            
        if contact_info:
            self.add_section("Contact Information", "\n".join(contact_info))
        
        # Add skills
        if result['skills']:
            skills_widget = QWidget()
            skills_layout = QVBoxLayout(skills_widget)
            
            # Group skills by category if available
            if isinstance(result['skills'], dict):
                for category, skills in result['skills'].items():
                    if skills:
                        cat_label = QLabel(category)
                        cat_label.setObjectName("sectionTitle")
                        skills_layout.addWidget(cat_label)
                        
                        tags_widget = QWidget()
                        tags_layout = FlowLayout()
                        for skill in skills:
                            tag = QLabel(skill)
                            tag.setObjectName("skillTag")
                            tags_layout.addWidget(tag)
                        tags_widget.setLayout(tags_layout)
                        skills_layout.addWidget(tags_widget)
            else:
                # Fallback for flat skill list
                tags_widget = QWidget()
                tags_layout = FlowLayout()
                for skill in result['skills']:
                    tag = QLabel(skill)
                    tag.setObjectName("skillTag")
                    tags_layout.addWidget(tag)
                tags_widget.setLayout(tags_layout)
                skills_layout.addWidget(tags_widget)
            
            self.scroll_layout.addWidget(skills_widget)
        
        # Scroll to top
        self.scroll_layout.itemAt(0).widget().scroll(0, 0)
        self.statusBar.showMessage("Resume processed successfully", 3000)
    
    def add_section(self, title, content):
        section_title = QLabel(title)
        section_title.setObjectName("sectionTitle")
        self.scroll_layout.addWidget(section_title)
        
        content_label = QLabel(content)
        content_label.setObjectName("infoText")
        content_label.setWordWrap(True)
        self.scroll_layout.addWidget(content_label)
        
        # Add separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.scroll_layout.addWidget(line)
    
    def show_error(self, error_msg):
        self.statusBar.showMessage(f"Error: {error_msg}", 5000)
    
    def worker_finished(self):
        self.upload_btn.setEnabled(True)
        self.parse_btn.setEnabled(True)
        self.parse_btn.setText("Parse Resume")

# Custom flow layout for skill tags
class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=0, spacing=-1):
        super().__init__(parent)
        self.itemList = []
        self.setSpacing(spacing)
        self.setContentsMargins(margin, margin, margin, margin)
    
    def addItem(self, item):
        self.itemList.append(item)
    
    def count(self):
        return len(self.itemList)
    
    def itemAt(self, index):
        if 0 <= index < len(self.itemList):
            return self.itemList[index]
        return None
    
    def takeAt(self, index):
        if 0 <= index < len(self.itemList):
            return self.itemList.pop(index)
        return None
    
    def expandingDirections(self):
        return Qt.Orientations(Qt.Orientation(0))
    
    def hasHeightForWidth(self):
        return True
    
    def heightForWidth(self, width):
        return self.doLayout(QRect(0, 0, width, 0), True)
    
    def setGeometry(self, rect):
        super().setGeometry(rect)
        self.doLayout(rect, False)
    
    def sizeHint(self):
        return self.minimumSize()
    
    def minimumSize(self):
        size = QSize()
        for item in self.itemList:
            size = size.expandedTo(item.minimumSize())
        margin = self.contentsMargins().left()
        size += QSize(2 * margin, 2 * margin)
        return size
    
    def doLayout(self, rect, testOnly):
        x = rect.x()
        y = rect.y()
        line_height = 0
        
        for item in self.itemList:
            space_x = self.spacing()
            space_y = self.spacing()
            next_x = x + item.sizeHint().width() + space_x
            if next_x - space_x > rect.right() and line_height > 0:
                x = rect.x()
                y = y + line_height + space_y
                next_x = x + item.sizeHint().width() + space_x
                line_height = 0
                
            if not testOnly:
                item.setGeometry(QRect(x, y, item.sizeHint().width(), item.sizeHint().height()))
                
            x = next_x
            line_height = max(line_height, item.sizeHint().height())
            
        return y + line_height - rect.y()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show the main window
    window = ResumeSkillExtractorApp()
    window.show()
    
    sys.exit(app.exec_())
