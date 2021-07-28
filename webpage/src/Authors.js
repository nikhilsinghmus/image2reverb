import React from "react";
import { Container, Row, Col } from "react-bootstrap";

const Authors = (props) => {
    let authors = [];
    let institutions = {};
    
    var i = 1;
    for (const author of props.authors) {
        const inst = author.institution.split(",").map((x) => x.trim());
        for (const institution of inst) {
            if (!institutions[institution]) {
                institutions[institution] = i;
                i++;
            }
        }
        authors.push(
            <React.Fragment>
                <a href={author.url}>{author.name}</a>
                <sup>{inst.map((institution) => institutions[institution]).join(",")}</sup><span>{(author == props.authors[props.authors.length - 1]) ? "" : ", "}</span>
            </React.Fragment>
        );
    }

    return (
        <Container style={{paddingBottom: 40, width: "100%"}} className="justify-content-md-left" fluid>
            <Row>
                <h6 align="center">{authors}</h6>
            </Row>
            <Row>
                <p align="center">{Object.entries(institutions).map(([inst, i], n) => (<React.Fragment><sup>{i}</sup>{inst}<span>{(n == (Object.keys(institutions).length - 1)) ? "" : ", "}</span></React.Fragment>))}</p>
            </Row>
            <Row>
                <p align="center"><b>{props.venue}</b></p>
            </Row>
            <Row>
                <h5 align="center"><a href={props.paper} target="_blank">[Paper]</a> <a href={props.github} target="_blank">[Github]</a></h5>
            </Row>
        </Container>
    );
}

export default Authors;
