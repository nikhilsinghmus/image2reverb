import { Container, Row, Col } from "react-bootstrap";

const GPalette = (props) => {
    var examples = [];
    for (var i = 0; i < 6; ++i) {
        if (props.palette[i]) {
            examples.push(props.palette[i]);
        }
    }

    const height = 120;

    const p = (x) => (<p align="center">{x}</p>);

    return (
        <Container style={{paddingBottom: 40, width: "100%"}} className="justify-content-md-left" fluid>
            <h4 align="center">{props.palette.category}</h4>
            <br/>
            <Row>
                <Col xs="auto">
                    
                {examples.map((example, i) => {
                    return (
                        <Row key={i}>
                            <Col xs="auto">{i == 0 ? p("Input") : null}<img src={require(props.src_dir + example + "/input.png").default} width={height} height={height}/></Col>
                            <Col xs="auto">{i == 0 ? p("Depth") : null}<img src={require(props.src_dir + example + "/depth.png").default} width={height} height={height}/></Col>
                            <Col xs="auto">{i == 0 ? p("Ground-Truth IR") : null}<video height={height} width={height} controls><source src={require(props.src_dir + example + "/gtir_spectrogram.mp4").default}/></video></Col>
                            <Col xs="auto">{i == 0 ? p("Output (Ours)") : null}<video height={height} width={height} controls><source src={require(props.src_dir + example + "/ir_spectrogram.mp4").default}/></video></Col>
                            <Col xs="auto">{i == 0 ? p("Ground-Truth Convolved") : null}<video height={height} width={height*2.5} controls><source src={require(props.src_dir + example + "/orig_spectrogram.mp4").default}/></video></Col>
                            <Col xs="auto">{i == 0 ? p("Convolved (Ours)") : null}<video height={height} width={height*2.5} controls><source src={require(props.src_dir + example + "/convolved_spectrogram.mp4").default}/></video></Col>
                        </Row>
                    )
                })}
                </Col>
            </Row>
            <br/>
        </Container>
    );
}

export default GPalette;
