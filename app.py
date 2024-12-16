from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__, static_folder="static")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run_tool():
    try:
        # Extract code and tool type from the request
        code = request.form.get('code', '')
        tool = request.form.get('tool', '')

        if not code.strip():
            return jsonify({"status": "error", "message": "No code provided."})

        # Run the selected tool
        if tool == 'lexer':
            lexer = Lexer(code)
            tokens = list(lexer.generate_tokens())
            token_output = "\n".join([str(token) for token in tokens])
            return jsonify({"status": "success", "output": token_output})

        elif tool == 'parser':
            lexer = Lexer(code)
            tokens = list(lexer.generate_tokens())
            parser = Parser(tokens)
            parser.parse()
            parser_output = "\n".join([message for message in parser.show_instructions()])
            return jsonify({"status": "success", "output": "Parsing completed successfully.\n" + parser_output})

        elif tool == 'semantic':
            lexer = Lexer(code)
            tokens = list(lexer.generate_tokens())
            parser = ParserWithSemantics(tokens)
            parser.parse()
            return jsonify({"status": "success", "output": "Semantic analysis completed successfully."})

        else:
            return jsonify({"status": "error", "message": "Invalid tool selected."})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded."})

        file = request.files['file']

        if not file.filename.endswith('.snk'):
            return jsonify({"status": "error", "message": "Invalid file type. Please upload a .snk file."})

        code = file.read().decode('utf-8')
        return jsonify({"status": "success", "code": code})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)