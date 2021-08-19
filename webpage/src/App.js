import React, { useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import { Container, Navbar, Col, Row, Dropdown, Button, ButtonGroup } from "react-bootstrap";
import Palette from "./Palette";
import GPalette from "./GPalette";
import Authors from "./Authors";
import Abstract from "./Abstract";
import authorlist from "./authorlist";
import metadata from "./metadata";
import palettes from "./palettes";

const App = () => {
    const g = {
        "0": "208_2_acoustics-and-psychoacoustics-book",
        "1": "263_8_acoustics-and-psychoacoustics-book",
        "2": "193_20_acoustics-and-psychoacoustics-book",
        "3": "212_80_acoustics-and-psychoacoustics-book",
        "category": "Examples with Ground Truth Impulse Responses",
        "order": -1,
        "signal": "acoustics-and-psychoacoustics-book"
    }

    const [currentPalette, setCurrentPalette] = useState(0);

    return (
        <Container fluid style={{width: "100%", paddingLeft: 0, paddingRight: 0}}>
            <Container fluid style={{width: "100%"}}>
                <br/><br/>
                <h2 align="center"><b>Image2Reverb</b>: Cross-Model Reverb Impulse Response Synthesis</h2>
                <Authors authors={authorlist} venue={metadata.venue} paper={metadata.paper} github={metadata.github}/>
            </Container>
            <Container fluid>
                <Abstract img={metadata.img} abstract={metadata.abstract} tldr={metadata.tldr}/>
                <Row align="center"><h4 style={{paddingTop: 40}}><b>BibTeX</b></h4></Row>
                <Row><pre style={{width: "80%", margin: "0 auto", backgroundColor: "rgba(240, 244, 248, 1)", padding: 20, borderRadius: 8, fontFamily: "monospace"}}>{metadata.bibtex}</pre></Row>
                <Col>
                    <Row align="center"><h4 style={{paddingTop: 40}}><b>Examples</b></h4></Row>
                    <Row align="center"><p>Audiovisual input/output examples accompanying the paper.</p></Row>
                </Col>
            </Container>
            <GPalette palette={g} src_dir={"./examples/"} key={g.order}/>
            <h6 align="center"><b>N.B.</b> the following examples generally do not have ground truth IRs available. Instead, these demonstrate various applications of our model.<br/><br/>Select a category from the dropdown menu to see the corresponding examples.</h6>
            <br/>
            <Dropdown as={ButtonGroup}>
                <Button variant="light" style={{backgroundColor: "white", borderColor: "white"}}><h4>{palettes[currentPalette].category}</h4></Button>
                <Dropdown.Toggle variant="light" size="lg" id="dropdown-split-basic" style={{backgroundColor: "white", borderColor: "white"}}/>
                <Dropdown.Menu>
                    {palettes.map((palette, i) => (
                        <Dropdown.Item eventKey={`${i}`} key={i} onClick={() => setCurrentPalette(i)}>{palette.category}</Dropdown.Item>
                    ))}
                </Dropdown.Menu>
            </Dropdown>
            <br/>
            <Palette key={currentPalette} palette={palettes[currentPalette]} src_dir={"./examples/"}/>
            <p align="center" style={{fontSize: 12, color: "gray"}}>Updated August 19th 2021.</p>
        </Container>
    );
}

export default App;